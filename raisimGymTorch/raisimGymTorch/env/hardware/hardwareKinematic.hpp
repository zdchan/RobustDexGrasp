#ifndef HARDWARE_KINEMATIC_HPP
#define HARDWARE_KINEMATIC_HPP

#include "Yaml.hpp"

// raisim library
#include "raisim/World.hpp"
#include "raisim/math.hpp"

class HardwareKinematic {
public:
    typedef enum ik_err_code {
        IK_OK = 0,
        IK_FAIL = 1,
        IK_TIMEOUT = 2,
        IK_SELF_COLLISION = 3
    } IK_ERRCODE;

    virtual void init(const std::string &rsc_pth, const Yaml::Node &cfg) = 0;
    virtual void setFrameVelocityNames(std::vector<std::string> &names) = 0;
    virtual IK_ERRCODE getArmIKSolve(const Eigen::VectorXd eef, const Eigen::VectorXd current_q, Eigen::VectorXd &solved_q) const = 0;
    virtual void setSimPlatform(raisim::ArticulatedSystem *platform) = 0;
    virtual void updateURDFFK(const Eigen::VectorXd &joint) = 0;
    virtual void getFrameOrientation(const std::string &jointName, const std::string &linkName, raisim::Mat<3, 3> &orientation_W) = 0;
    virtual void getFramePosition(const std::string &jointName, const std::string &linkName, raisim::Vec<3> &point_W) = 0;
    virtual void getFrameAngularVelocity(const std::string &frameName, raisim::Vec<3> &angVel_W) = 0;
    virtual void getFrameVelocity(const std::string &frameName, raisim::Vec<3> &vel_W) = 0;
};


#endif //HARDWARE_KINEMATIC_HPP
