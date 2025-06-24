#include "../hardwareArm.hpp"

// raisim library
#include "raisim/World.hpp"
#include "raisim/math.hpp"
#include <time.h> 

class UR5Sim : public HardwareArm {
public:
    void init(const std::string &rsc_pth, const Yaml::Node &cfg) final override {
        if (!cfg["randomize_gc_arm"].IsNone()) {
            randomize_gc_ = cfg["randomize_gc_arm"].As<double>();
        }
        std::string pd_file = cfg["arm_pd_file"].As<std::string>();

        arm_joint_position_.setZero(num_joint_);
        arm_joint_velocity_.setZero(num_joint_);
        end_effector_pose_.setZero(6);
        end_effector_velocity_.setZero(3);
        end_effector_angle_velocity_.setZero(3);
        arm_init_base_pose_.setZero(6);
        arm_init_base_pose_ << 0.55, 0.75152, 0.0, 0.0, 0.0, 0.0;

        std::ifstream pd_txt;
        pd_txt.open(rsc_pth+"/../raisimGymTorch/raisimGymTorch/env/hardware/arm/"+pd_file);
        if (pd_txt) {
            std::string line;
            int line_cnt = 0;
            while (getline(pd_txt, line)) {
                std::stringstream ss(line);
                if (line_cnt < num_joint_) {
                    ss >> Pgain[line_cnt];
                } else {
                    ss >> Dgain[line_cnt - num_joint_];
                }
                line_cnt++;
            }
            if (line_cnt != (num_joint_*2)) {
                std::cout << "error txt line:" << line_cnt << std::endl;
                pd_txt.close();
                exit(0);
            }
            pd_txt.close();
        } else {
            for (int i = 0; i < num_joint_; i++) {
                Pgain[i] = Pgain[0];
                Dgain[i] = Dgain[0];
            }
        }

        srand(time(0));
    }
    void setSimPlatform(raisim::ArticulatedSystem *platform) final override {
        platform_ = platform;
        std::vector<raisim::Vec<2>> joint_limits = platform_->getJointLimits();
        joint_limit_high_.setZero(num_joint_); joint_limit_low_.setZero(num_joint_);
        for(int i=0; i < num_joint_; i++){
            joint_limit_low_[i] = joint_limits[i][0];
            joint_limit_high_[i] = joint_limits[i][1];
        }
    }

    void updateArmState() final override {
        int gc_dim = platform_->getGeneralizedCoordinateDim();
        int gv_dim = platform_->getDOF();
        Eigen::VectorXd gc(gc_dim), gv(gv_dim);
        platform_->getState(gc, gv);
        arm_joint_position_ = gc.head(num_joint_);
        arm_joint_velocity_ = gv.head(num_joint_);
        if (randomize_gc_ > 1e-9) {
            arm_joint_position_ += Eigen::VectorXd::Random(num_joint_) * randomize_gc_;
            arm_joint_position_ = arm_joint_position_.cwiseMax(joint_limit_low_).cwiseMin(joint_limit_high_);
        }

        raisim::Mat<3,3> eef_rot;
        platform_->getFrameOrientation("Flange2hand_fixed_joint", eef_rot);
        raisim::Vec<3> eef_eul;
        raisim::RotmatToEuler(eef_rot, eef_eul);
        raisim::Vec<3> eef_pos;
        platform_->getFramePosition("Flange2hand_fixed_joint", eef_pos);
        eef_pos[0] -= 0.55;
        eef_pos[1] -= 0.75152;
        eef_pos[2] -= 0.771;
        end_effector_pose_.head(3) = eef_pos.e();
        end_effector_pose_.tail(3) = eef_eul.e();

        raisim::Vec<3> eef_vel, eef_angle_vel;
        platform_->getFrameVelocity("Flange2hand_fixed_joint", eef_vel);
        platform_->getFrameAngularVelocity("Flange2hand_fixed_joint", eef_angle_vel);
        end_effector_velocity_ = eef_vel.e();
        end_effector_angle_velocity_ = eef_angle_vel.e();
    }

    void setPdTarget(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget, bool async = true) const final override {
        platform_->setPdTarget(posTarget, velTarget);
    }

    void getPdgains(Eigen::VectorXd &pgain, Eigen::VectorXd &dgain, int head_shift) const final override {
        for (int i = 0; i < num_joint_; i++) {
            pgain[i] = Pgain[i];
            dgain[i] = Dgain[i];
//            std::cout << "UR5 Sim joint[" << i << "] P=" << pgain[i] << ", D=" << dgain[i] << std::endl;
        }
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

private:
    raisim::ArticulatedSystem *platform_;

    double randomize_gc_ = 0.0;
    Eigen::VectorXd joint_limit_high_;
    Eigen::VectorXd joint_limit_low_;

    const static int num_joint_ = 6;

    double Pgain[num_joint_] = {3000.0};
    double Dgain[num_joint_] = {150};

    const std::string body_parts_[6] =  {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
    const std::string contact_bodies_[6] =  {"shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"};
};

extern "C" std::unique_ptr<HardwareArm> createUR5Sim() {
    return std::make_unique<UR5Sim>();
}