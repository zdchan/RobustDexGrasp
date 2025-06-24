#include "../hardwareArm.hpp"

// raisim library
#include "raisim/World.hpp"
#include "raisim/math.hpp"

class FlyingSim : public HardwareArm {
public:
    void init(const std::string &rsc_pth, const Yaml::Node &cfg) final override {
        arm_joint_position_.setZero(num_joint_);
        arm_joint_velocity_.setZero(num_joint_);
        end_effector_pose_.setZero(6);
        end_effector_velocity_.setZero(3);
        end_effector_angle_velocity_.setZero(3);
        arm_init_base_pose_.setZero(6);
        arm_init_base_pose_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    }
    void setSimPlatform(raisim::ArticulatedSystem *platform) final override {
        platform_ = platform;
    }

    void updateArmState() final override {
        int gc_dim = platform_->getGeneralizedCoordinateDim();
        int gv_dim = platform_->getDOF();
        Eigen::VectorXd gc(gc_dim), gv(gv_dim);
        platform_->getState(gc, gv);
        arm_joint_position_ = gc.head(num_joint_);
        arm_joint_velocity_ = gv.head(num_joint_);

        raisim::Mat<3,3> eef_rot;
        platform_->getFrameOrientation("z_rotation_joint", eef_rot);
        raisim::Vec<3> eef_eul;
        raisim::RotmatToEuler(eef_rot, eef_eul);
        raisim::Vec<3> eef_pos;
        platform_->getFramePosition("z_rotation_joint", eef_pos);
        end_effector_pose_.head(3) = eef_pos.e();
        end_effector_pose_.tail(3) = eef_eul.e();

        raisim::Vec<3> eef_vel, eef_angle_vel;
        platform_->getFrameVelocity("z_rotation_joint", eef_vel);
        platform_->getFrameAngularVelocity("z_rotation_joint", eef_angle_vel);
        end_effector_velocity_ = eef_vel.e();
        end_effector_angle_velocity_ = eef_angle_vel.e();
    }

    void setPdTarget(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget, bool async = true) const final override {
        platform_->setPdTarget(posTarget, velTarget);
    }

    void getPdgains(Eigen::VectorXd &pgain, Eigen::VectorXd &dgain, int head_shift) const final override {
        pgain.head(3).setConstant(Pgain);    // xyz joint of wrist
        dgain.head(3).setConstant(Dgain);
        pgain.segment(3, 3).setConstant(Pgain_rot);   // rxryrz joint of wrist
        dgain.segment(3, 3).setConstant(Dgain_rot);
    }

    const int getDim() const final override {
        return num_joint_;
    }

    Eigen::VectorXd & getSimBasePose() final override {
        return arm_init_base_pose_;
    }

    Eigen::VectorXd & getJointVelocity() final override {
        return arm_joint_velocity_;
    }
    Eigen::VectorXd & getJointPosition() final override {
        return arm_joint_position_;
    }
    Eigen::VectorXd & getEefPose() final override {
        return end_effector_pose_;
    }
    Eigen::VectorXd & getEefVelocity() final override {
        return end_effector_velocity_;
    }
    Eigen::VectorXd & getEefAngleVelocity() final override {
        return end_effector_angle_velocity_;
    }
    int getBodies(std::vector<std::string> & get_vec, bool contact_flag) const final override {
        if (contact_flag) {
            for (int i = 0; i < 6; i++) {
                get_vec.push_back(contact_bodies_[i]);
            }
            return 6;
        } else {
            for (int i = 0; i < 6; i++) {
                get_vec.push_back(body_parts_[i]);
            }
            return 6;
        }
    }

private:
    raisim::ArticulatedSystem *platform_;

    const double Pgain = 100.0;
    const double Dgain = 0.1;
    const double Pgain_rot = 100.0;
    const double Dgain_rot = 0.2;

    const int num_joint_ = 6;
    
    const std::string body_parts_[1] =  {"None"};
    const std::string contact_bodies_[1] =  {"None"};
};

extern "C" std::unique_ptr<HardwareArm> createFlyingSim() {
    return std::make_unique<FlyingSim>();
}
