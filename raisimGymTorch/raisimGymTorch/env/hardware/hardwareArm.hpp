#ifndef HARDWARE_ARM_HPP
#define HARDWARE_ARM_HPP

#include "Yaml.hpp"

// raisim library
#include "raisim/World.hpp"
#include "raisim/math.hpp"

class HardwareArm {
public:
    virtual void init(const std::string &rsc_pth, const Yaml::Node &cfg) = 0;
    virtual void setSimPlatform(raisim::ArticulatedSystem *platform) = 0; // only use in simulation mode

    virtual void updateArmState() = 0;
    virtual void setPdTarget(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget, bool async = true) const = 0;

    virtual void getPdgains(Eigen::VectorXd &pgain, Eigen::VectorXd &dgain, int head_shift) const = 0;
    virtual const int getDim() const = 0;
    
    virtual Eigen::VectorXd & getSimBasePose() = 0;
    virtual Eigen::VectorXd & getJointPosition() = 0;
    virtual Eigen::VectorXd & getJointVelocity() = 0;
    virtual Eigen::VectorXd & getEefPose() = 0;
    virtual Eigen::VectorXd & getEefVelocity() = 0;
    virtual Eigen::VectorXd & getEefAngleVelocity() = 0;
    virtual int getBodies(std::vector<std::string> & get_vec, bool contact_flag) const = 0;

    virtual ~HardwareArm() = default;

    // value updated after using updateArmState()
    // x,y,z,rx,ry,rz: the position(m) and euler angle(rad) of the eef frame expressed in base frame 
    Eigen::VectorXd end_effector_pose_;
    // vx,vy,vz: the linear velocity(m/s) of the eef frame expressed in the base frame
    Eigen::VectorXd end_effector_velocity_;
    // wx,wy,wz: the angular velocity(rad/s) of the eef frame expressed in the base frame
    Eigen::VectorXd end_effector_angle_velocity_;
    // p0~p5: the position(rad) of each revolute joint describe in URDF, or prismatic joint for flying
    Eigen::VectorXd arm_joint_position_;
    // p0~p5: the velocity(rad/s) of each revolute joint describe in URDF, or prismatic joint for flying
    Eigen::VectorXd arm_joint_velocity_;
    // x,y,z,rx,ry,rz: the position(m) and euler angle(rad) of the base frame expressed in raisim wrold frame 
    Eigen::VectorXd arm_init_base_pose_;
};

#endif //HARDWARE_ARM_HPP
