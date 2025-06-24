#ifndef HARDWARE_HAND_HPP
#define HARDWARE_HAND_HPP

#include "Yaml.hpp"

// raisim library
#include "raisim/World.hpp"
#include "raisim/math.hpp"

class HardwareHand {
public:
    // load urdf and set the configuration of this arm
    virtual void init(const std::string &rsc_pth, const Yaml::Node &cfg) = 0;
    virtual void setSimPlatform(raisim::ArticulatedSystem *platform) = 0; // only use in simulation mode

    virtual void updateHandState(const Eigen::VectorXd &eef_pos) = 0;
    virtual void setPdTarget(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget, bool async = true) = 0;

    virtual void getPdgains(Eigen::VectorXd &pgain, Eigen::VectorXd &dgain, int tail_shift) const = 0;
    virtual Eigen::VectorXd & getJointPosition() = 0;
    virtual Eigen::VectorXd & getJointVelocity() = 0;
    virtual const int getDim() const = 0;
    virtual const int getNumFinger() const = 0;
    virtual int getBodies(std::vector<std::string> & get_vec, bool contact_flag) const = 0;
    virtual std::string changeJointToLinkName(std::string frameName) const = 0;

    virtual ~HardwareHand() = default;

    // value updated after using updateHandState()
    std::vector<raisim::Mat<3, 3>> frame_orientation_;
    std::vector<raisim::Vec<3>> frame_position_;
    // wrist pose in world frame
    Eigen::VectorXd wrist_pose_;
    Eigen::VectorXd wrist_velocity_;
    Eigen::VectorXd hand_joint_position_;
    Eigen::VectorXd hand_joint_velocity_;
};


#endif //HARDWARE_HAND_HPP
