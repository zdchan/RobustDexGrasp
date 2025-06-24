#include "../hardwareKinematic.hpp"

// cpp library
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <time.h>

class SimFK : public HardwareKinematic {
public:
    void init(const std::string &rsc_pth, const Yaml::Node &cfg) override {
        if (!cfg["randomize_frame_position"].IsNone()) {
            randomize_frame_position_ = cfg["randomize_frame_position"].As<bool>();
        }
        if (!cfg["randomize_frame_orientation"].IsNone()) {
            randomize_frame_orientation_ = cfg["randomize_frame_orientation"].As<double>();
        }

        srand(time(0));
    }

    void setFrameVelocityNames(std::vector<std::string> &names) override {
    }

    IK_ERRCODE getArmIKSolve(const Eigen::VectorXd eef, const Eigen::VectorXd current_q, Eigen::VectorXd &solved_q) const override {
    }

    void setSimPlatform(raisim::ArticulatedSystem *platform) final override {
        platform_ = platform;
    }

    void updateURDFFK(const Eigen::VectorXd &joint) override {
    }

    void getFrameOrientation(const std::string &jointName, const std::string &linkName, raisim::Mat<3, 3> &orientation_W) final override {
        platform_->getFrameOrientation(jointName, orientation_W);
        if (randomize_frame_orientation_ > 1e-9) {
            raisim::Vec<3> random_euler; 
            random_euler.e() = Eigen::VectorXd::Random(3) * randomize_frame_orientation_;
            raisim::Vec<4> quat_temp;
            raisim::Mat<3, 3> random_mat;
            raisim::eulerToQuat(random_euler, quat_temp);
            raisim::quatToRotMat(quat_temp, random_mat);
            raisim::Mat<3, 3> orientation_new;
            raisim::matmul(orientation_W, random_mat, orientation_new);
            orientation_W = orientation_new;

        }
    }
    void getFramePosition(const std::string &jointName, const std::string &linkName, raisim::Vec<3> &point_W) final override {
        platform_->getFramePosition(jointName, point_W);
        if (randomize_frame_position_ > 1e-9) {
            point_W.e() += Eigen::VectorXd::Random(3) * randomize_frame_position_;
        }
    }
    void getFrameAngularVelocity(const std::string &frameName, raisim::Vec<3> &angVel_W) final override {
        platform_->getFrameAngularVelocity(frameName, angVel_W);
    }
    void getFrameVelocity(const std::string &frameName, raisim::Vec<3> &vel_W) final override {
        platform_->getFrameVelocity(frameName, vel_W);
    }

private:
    raisim::ArticulatedSystem *platform_;

    double randomize_frame_orientation_ = 0.0;
    double randomize_frame_position_ = 0.0;
};

extern "C" std::unique_ptr<HardwareKinematic> createSimFK() {
    return std::make_unique<SimFK>();
}
