#include "../hardwareHand.hpp"

// raisim library
#include "raisim/World.hpp"
#include "raisim/math.hpp"

class LeapReal : public HardwareHand {
public:
    void init(const std::string &rsc_pth, const Yaml::Node &cfg) final override {
        wrist_pose_.setZero(6);
        wrist_velocity_.setZero(6);
        hand_joint_position_.setZero(num_joint_);
        hand_joint_velocity_.setZero(num_joint_);
        
        flying_hand_mode_ = cfg["flying_hand_mode"].As<bool>();
    }
    void setSimPlatform(raisim::ArticulatedSystem *platform) final override {
        platform_ = platform;
    }

    void updateHandState(const Eigen::VectorXd &eef_pos) final override {
        std::cout << "updateHandState in real Leap: TBD" << std::endl;

    }
    void setPdTarget(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget, bool async = true) final override {
        std::cout << "setPdTarget in real Leap: TBD" << std::endl;
    }

    void getPdgains(Eigen::VectorXd &pgain, Eigen::VectorXd &dgain, int tail_shift) const final override {
        pgain.tail(tail_shift).setConstant(Pgain);
        dgain.tail(tail_shift).setConstant(Dgain);
    }

    Eigen::VectorXd & getJointVelocity() final override {
        return hand_joint_velocity_;
    }
    Eigen::VectorXd & getJointPosition() final override {
        return hand_joint_position_;
    }
    const int getDim() const final override {
        return num_joint_;
    }
    const int getNumFinger() const final override {
        return num_finger_;
    }
    int getBodies(std::vector<std::string> & get_vec, bool contact_flag) const final override {
        if (contact_flag) {
            for (int i = 0; i < num_contacts_; i++) {
                get_vec.push_back(contact_bodies_[i]);
            }
            return num_contacts_;
        } else {
            for (int i = 0; i < num_bodies_; i++) {
                get_vec.push_back(body_parts_[i]);
            }
            return num_bodies_;
        }
    }

    std::string changeJointToLinkName(std::string frameName) const final override {
        return frameName;
    }

private:
    raisim::ArticulatedSystem *platform_;

    bool flying_hand_mode_ = false;

    const static int num_contacts_ = 13;
    const static int num_bodies_ = 17;
    const static int num_finger_ = 4;
    const static int num_joint_ = 16;

    const double Pgain = 60.0;
    const double Dgain = 0.2;

    const std::string body_parts_flying_[num_bodies_] = {"TBD"};

    const std::string body_parts_[num_bodies_] =  {"wrist_3_link-tool0_fixed_joint",
    "leap_joint1", "leap_joint2", "leap_joint3", "leap_joint3_tip",
    "leap_joint5", "leap_joint6", "leap_joint7", "leap_joint7_tip",
    "leap_joint9", "leap_joint10", "leap_joint11", "leap_joint11_tip",
    "leap_joint13", "leap_joint14", "leap_joint15", "leap_joint15_tip"};

    // for raisim contact check
    const std::string contact_bodies_[num_contacts_] =   {"wrist_3_link",
    "pip", "dip", "fingertip",
    "pip_2", "dip_2", "fingertip_2",
    "pip_3", "dip_3", "fingertip_3",
    "pip_4", "thumb_dip", "thumb_fingertip"};
};

extern "C" std::unique_ptr<HardwareHand> createLeapReal() {
    return std::make_unique<LeapReal>();
}
