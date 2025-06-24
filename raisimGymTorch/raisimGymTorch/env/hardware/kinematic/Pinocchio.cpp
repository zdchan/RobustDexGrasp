#include "../hardwareKinematic.hpp"

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/srdf.hpp"
 
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
 
#include "pinocchio/multibody/sample-models.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/jacobian.hpp"

#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/collision/collision.hpp"

// cpp library
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <time.h>

class Pinocchio : public HardwareKinematic {
public:
    void init(const std::string &rsc_pth, const Yaml::Node &cfg) override {
        const std::string pth = rsc_pth + "/" + cfg["rsc_model"].As<std::string>();
        const std::string fk_hand_urdf_pth = pth + "/" + cfg["kinematic_real"]["fk_model"].As<std::string>() + ".urdf";
        const std::string ik_arm_urdf_pth = pth + "/" + cfg["kinematic_real"]["ik_model"].As<std::string>() + ".urdf";
        const std::string ik_arm_srdf_pth = pth + "/" + cfg["kinematic_real"]["ik_model"].As<std::string>() + ".srdf";
        flying_mode_ = cfg["flying_hand_mode"].As<bool>();
        velocity_dt_s_ = cfg["kinematic_real"]["real_velocity_dt_s"].As<double>();

        hand_fk_model_ = std::make_unique<pinocchio::Model>();
        pinocchio::urdf::buildModel(fk_hand_urdf_pth, *hand_fk_model_);
        hand_fk_data_ = std::make_unique<pinocchio::Data>(*hand_fk_model_);

        if (flying_mode_ == false) {
            arm_ik_model_ = std::make_unique<pinocchio::Model>();
            pinocchio::urdf::buildModel(ik_arm_urdf_pth, *arm_ik_model_);
            arm_ik_data_ = std::make_unique<pinocchio::Data>(*arm_ik_model_);
    
            arm_ik_geom_model_ = std::make_unique<pinocchio::GeometryModel>();
            pinocchio::urdf::buildGeom(*arm_ik_model_, ik_arm_urdf_pth, pinocchio::COLLISION, *arm_ik_geom_model_, pth);
            arm_ik_geom_model_->addAllCollisionPairs();
            pinocchio::srdf::removeCollisionPairs(*arm_ik_model_, *arm_ik_geom_model_, ik_arm_srdf_pth);
            arm_ik_geom_data_ = std::make_unique<pinocchio::GeometryData>(*arm_ik_geom_model_);
            pinocchio::srdf::loadReferenceConfigurations(*arm_ik_model_, ik_arm_srdf_pth); 
        }

        std::cout << "------------- pinocchio real init finish !!!!!" << std::endl;
    }

    void setFrameVelocityNames(std::vector<std::string> &names) override {
        for (int i = 0; i < names.size(); i++) {
            vel_id_.push_back(hand_fk_model_->getBodyId(names[i]));
        }
        last_R_.resize(names.size());
        last_t_.resize(names.size());
        diff_R_.resize(names.size());
        diff_t_.resize(names.size());
    }

    // eef is x, y, z(m), rx, ry, rz(rad)
    // current_q is current joint position
    IK_ERRCODE getArmIKSolve(const Eigen::VectorXd eef, const Eigen::VectorXd current_q, Eigen::VectorXd &solved_q) const override {
        IK_ERRCODE ik_flag = IK_FAIL;
        if (flying_mode_) {
            return IK_FAIL;
        }

        Eigen::Matrix3d rot;
        rot = Eigen::AngleAxisd(eef[3], Eigen::Vector3d::UnitX()) * 
                        Eigen::AngleAxisd(eef[4], Eigen::Vector3d::UnitY()) * 
                        Eigen::AngleAxisd(eef[5], Eigen::Vector3d::UnitZ());
        pinocchio::SE3 target_SE3(rot, Eigen::Vector3d(eef[0], eef[1], eef[2]));

        Eigen::VectorXd q = current_q; 
        pinocchio::FrameIndex frame_id = arm_ik_model_->getFrameId("tool0");

        pinocchio::Data::Matrix6x J(6, arm_ik_model_->nv);
        J.setZero();

        Eigen::Matrix<double, 6, 1> err;
        Eigen::VectorXd v(arm_ik_model_->nv);

        int timeout_cnt = 0;
        int step = 0;

        while (1) {
            pinocchio::framesForwardKinematics(*arm_ik_model_, *arm_ik_data_, q);
            const pinocchio::SE3 iMd = arm_ik_data_->oMf[frame_id].actInv(target_SE3); // A X B^-1
            err = pinocchio::log6(iMd).toVector(); // in joint frame

            // reach error range and break
            if (err.norm() < eps) {
                ik_flag = IK_OK;
                break;
            }

            // The number of iterations exceeded the maximum, will init the q again from random_cfg or random_num
            if (step >= IT_MAX) {
                timeout_cnt++;
                step = 0;
                if (timeout_cnt < TMO_MAX / 2) {
                    q = pinocchio::randomConfiguration(*arm_ik_model_);
                } else {
                    for (int i = 0; i < 6; i++) {
                        q[i] = (rand()/double(RAND_MAX) - 0.5) * 2.0 * EIGEN_PI;
                    }
                }
                continue;
            }

            // try more then max times.
            if (timeout_cnt >= TMO_MAX) {
                timeout_cnt = -1;
                //std::cout << "err = " << err.transpose() << std::endl;
                ik_flag = IK_TIMEOUT;
                break;
            }
            
            pinocchio::computeFrameJacobian(*arm_ik_model_, *arm_ik_data_, q, frame_id, J); // J in joint frame
            pinocchio::Data::Matrix6 Jlog;
            pinocchio::Jlog6(iMd.inverse(), Jlog);
            J = -Jlog * J;
            pinocchio::Data::Matrix6 JJt;
            JJt.noalias() = J * J.transpose();
            JJt.diagonal().array() += damp;
            v.noalias() = -J.transpose() * JJt.ldlt().solve(err);
            q = pinocchio::integrate(*arm_ik_model_, q, v * DT);

            step++;
        }

        for(int i = 0; i < 6; i++) {
            while(q(i) < -M_PI) {
                q(i) += 2 * M_PI;
            }
            while(q(i) > M_PI) {
                q(i) -= 2 * M_PI;
            }
        }

        solved_q = q;

        if (ik_flag != IK_OK) {
            return ik_flag;
        }

        // check self collision
        if (false == pinocchio::computeCollisions(*arm_ik_model_, *arm_ik_data_, *arm_ik_geom_model_, *arm_ik_geom_data_, solved_q, true)) {
            return IK_OK;
        }

        for (size_t k = 0; k < arm_ik_geom_model_->collisionPairs.size(); ++k) {
            const pinocchio::CollisionPair & cp = arm_ik_geom_model_->collisionPairs[k];
            const hpp::fcl::CollisionResult & cr = arm_ik_geom_data_->collisionResults[k];
            if (cr.isCollision()) {
                std::cout << "collision pair: " << cp.first << " , " << cp.second << " - collision: " << std::endl;
                return IK_SELF_COLLISION;
            }
        }
        
        std::cout << "!!!!!! UNKNOW collision !!!!!!!!!" << std::endl;
        return IK_SELF_COLLISION;
    }

    void setSimPlatform(raisim::ArticulatedSystem *platform) final override {
    }

    void updateURDFFK(const Eigen::VectorXd &joint) override {
        pinocchio::framesForwardKinematics(*hand_fk_model_, *hand_fk_data_, joint);
        
        auto now_time = std::chrono::system_clock::now();
        diff_time_s_ = ((now_time - last_time_).count() / 1e9);
        
        // calculate average velocity
        if (diff_time_s_ > velocity_dt_s_) {
            if (diff_time_s_ < 2.0) {
                calculate_velocity_flag_ = true;
                for (int i = 0; i < vel_id_.size(); i++) {
                    int id = vel_id_[i];
                    diff_R_[i] = hand_fk_data_->oMf[id].rotation() * last_R_[i].transpose();
                    diff_t_[i] = -diff_R_[i] * last_t_[i] + hand_fk_data_->oMf[id].translation();
                }
            } else {
                //std::cout << "first init or sth. block dt=" << diff_time_s_ << std::endl;
                calculate_velocity_flag_ = false;
            }

            last_time_ = now_time;
            for (int i = 0; i < vel_id_.size(); i++) {
                int id = vel_id_[i];
                last_R_[i] = hand_fk_data_->oMf[id].rotation();
                last_t_[i] = hand_fk_data_->oMf[id].translation();
            }
        }
    }
    void getFrameOrientation(const std::string &jointName, const std::string &linkName, raisim::Mat<3, 3> &orientation_W) final override {
        int id = hand_fk_model_->getJointId(jointName);
        if (id == hand_fk_model_->njoints) {
            id = hand_fk_model_->getBodyId(linkName);
            orientation_W.e() = hand_fk_data_->oMf[id].rotation();
        } else {
            orientation_W.e() = hand_fk_data_->oMi[id].rotation();
        }
    }
    void getFramePosition(const std::string &jointName, const std::string &linkName, raisim::Vec<3> &point_W) final override {
        int id = hand_fk_model_->getJointId(jointName);
        if (id == hand_fk_model_->njoints) {
            id = hand_fk_model_->getBodyId(linkName);
            point_W.e() = hand_fk_data_->oMf[id].translation();
        } else {
            point_W.e() = hand_fk_data_->oMi[id].translation();
        }
    }
    void getFrameAngularVelocity(const std::string &frameName, raisim::Vec<3> &angVel_W) final override {
        if (calculate_velocity_flag_) {
            for (int i = 0; i < vel_id_.size(); i++) {
                if (hand_fk_model_->getBodyId(frameName) == vel_id_[i]) {
                    angVel_W = diff_R_[i].eulerAngles(0,1,2) / diff_time_s_;
                }
            }
        } else {
            angVel_W.setZero();
        }
    }
    void getFrameVelocity(const std::string &frameName, raisim::Vec<3> &vel_W) final override {
        if (calculate_velocity_flag_) {
            for (int i = 0; i < vel_id_.size(); i++) {
                if (hand_fk_model_->getBodyId(frameName) == vel_id_[i]) {
                    vel_W.e() = diff_t_[hand_fk_model_->getBodyId(frameName)] / diff_time_s_;
                }
            }
        } else {
            vel_W.setZero();
        }
    }

private:
    std::unique_ptr<pinocchio::Model> hand_fk_model_;
    std::unique_ptr<pinocchio::Data> hand_fk_data_;

    std::unique_ptr<pinocchio::Model> arm_ik_model_;
    std::unique_ptr<pinocchio::Data> arm_ik_data_;
    std::unique_ptr<pinocchio::GeometryModel> arm_ik_geom_model_;
    std::unique_ptr<pinocchio::GeometryData> arm_ik_geom_data_;

    bool flying_mode_ = false;

    const double eps =  0.005;  // desired position precision  0.01m 3degree
    const int IT_MAX = 1000;  // maximum number of iterations 
    const double DT = 0.1; // convergence rate (smaller may have higher resolution?)
    const double damp = 1e-6; // damping factor for the pseudoinversion
    const int TMO_MAX = 30; // maximum number of the cnt of over iterations
    
    std::vector<int> vel_id_;
    double velocity_dt_s_ = 0.0;
    double diff_time_s_ = 0.0;
    bool calculate_velocity_flag_ = false;
    std::chrono::system_clock::time_point last_time_;
    std::vector<Eigen::Matrix3d> last_R_;
    std::vector<Eigen::Vector3d> last_t_;
    std::vector<Eigen::Matrix3d> diff_R_;
    std::vector<Eigen::Vector3d> diff_t_;
};

extern "C" std::unique_ptr<HardwareKinematic> createPinocchio() {
    return std::make_unique<Pinocchio>();
}
