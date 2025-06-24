#include "../hardwareArm.hpp"

#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <ur_rtde/rtde_io_interface.h>

#include <thread>
#include <chrono>
#include <time.h>

class UR5Real : public HardwareArm {
public:
    void init(const std::string &rsc_pth, const Yaml::Node &cfg) final override {
        arm_joint_position_.setZero(num_joint_);
        arm_joint_velocity_.setZero(num_joint_);
        end_effector_pose_.setZero(6);
        end_effector_velocity_.setZero(3);
        end_effector_angle_velocity_.setZero(3);
        arm_init_base_pose_.setZero(6);
        last_end_effector_pose_.setZero(6);
        last_arm_joint_position_.setZero(num_joint_);
        arm_init_base_pose_ << 0.55, 0.75152, 0.0, 0.0, 0.0, 0.0;

        std::string robot_ip = cfg["arm_real"]["ip"].As<std::string>();
        double rtde_frequency = cfg["arm_real"]["freq_hz"].As<double>();
        control_dt_ = 1.0 / rtde_frequency;
        uint16_t flags = ur_rtde::RTDEControlInterface::FLAG_USE_EXT_UR_CAP;

        rtde_control_ = std::make_unique<ur_rtde::RTDEControlInterface>(robot_ip, rtde_frequency, flags);
        rtde_receive_ = std::make_unique<ur_rtde::RTDEReceiveInterface>(robot_ip, rtde_frequency);

        move_vel_ = cfg["arm_real"]["move_vel"].As<double>();
        move_acc_ = cfg["arm_real"]["move_acc"].As<double>();
        velocity_dt_s_ = cfg["arm_real"]["real_velocity_dt_s"].As<double>();
        servoJ_ahead_time_ = cfg["arm_real"]["servoJ_ahead_time"].As<double>();
        servoJ_gain_ = cfg["arm_real"]["servoJ_gain"].As<double>();
        std::string pd_file = cfg["arm_pd_file"].As<std::string>();

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
        std::cout << "------------- ur5 real init finish !!!!!" << std::endl;
    }
    void setSimPlatform(raisim::ArticulatedSystem *platform) final override {
        platform_ = platform;
    }
    void updateArmState() final override {
        auto now_time = std::chrono::system_clock::now();
        double diff_time_s = ((now_time - last_time_).count() / 1e9);

        // Actual Cartesian coordinates of the tool: (x,y,z,rx,ry,rz), in m and rad
        // where rx, ry and rz is a rotation vector representation of the tool orientation
        std::vector<double> actual_tcp_pose = rtde_receive_->getActualTCPPose();
        Eigen::Vector3d vec(actual_tcp_pose[3], actual_tcp_pose[4], actual_tcp_pose[5]);
        Eigen::AngleAxisd rotation_vector (vec.norm(), vec.normalized());
        Eigen::Vector3d eulerAngle = rotation_vector.matrix().eulerAngles(0,1,2);
        for (int i = 0; i < 3; i++) {
            end_effector_pose_[i] = actual_tcp_pose[i];
            end_effector_pose_[i+3] = eulerAngle[i];
        }

        // Actual joint positions in rad
        std::vector<double> joint_positions = rtde_receive_->getActualQ();
        for (int i = 0; i < 6; i++) {
            arm_joint_position_[i] = joint_positions[i];
        }

        // Actual joint speed in rad/s
        std::vector<double> actual_joint_speed = rtde_receive_->getActualQd();
        for (int i = 0; i < 6; i++) {
            arm_joint_velocity_[i] = actual_joint_speed[i];
        }

        // Actual speed of the tool given in Cartesian coordinates
        //std::vector<double> actual_tcp_speed = rtde_receive_->getActualTCPSpeed();
        //for (int i = 0; i < 3; i++) {
        //    end_effector_velocity_[i] = actual_tcp_speed[i];
        //}
    }
    void setPdTarget(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget, bool async = true) const final override {
        std::vector<double> tar_joint_pos;
        for (int i = 0; i < 6; i++) {
            tar_joint_pos.push_back(posTarget[i]);
        }

        if (async == false) {
            rtde_control_->servoStop();
            rtde_control_->stopScript();
            usleep(50000);
            rtde_control_->moveJ(tar_joint_pos, 0.5, 0.5);
            usleep(50000);
            rtde_control_->stopJ();
            rtde_control_->stopScript();
        } else {
            rtde_control_->servoJ(tar_joint_pos, 0, 0, control_dt_, servoJ_ahead_time_, servoJ_gain_);
        }
    }

    void getPdgains(Eigen::VectorXd &pgain, Eigen::VectorXd &dgain, int head_shift) const final override {
        for (int i = 0; i < num_joint_; i++) {
            pgain[i] = Pgain[i];
            dgain[i] = Dgain[i];
            std::cout << "UR5 Sim joint[" << i << "] P=" << pgain[i] << ", D=" << dgain[i] << std::endl;
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
    double const_angle(double in) {
        double out = in;
        while (out > M_PI) {
            out -= 2*M_PI;
        }
        while (out < -M_PI) {
            out += 2*M_PI;
        }
        return out;
    }

private:
    raisim::ArticulatedSystem *platform_;

    const static int num_joint_ = 6;

    double Pgain[num_joint_] = {3000.0};
    double Dgain[num_joint_] = {150.0};

    const std::string body_parts_[6] =  {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
    const std::string contact_bodies_[6] =  {"shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"};

    double move_vel_ = 0.0;
    double move_acc_ = 0.0;
    double velocity_dt_s_ = 0.0;
    double control_dt_ = 0.0;
    double servoJ_gain_ = 0.0;
    double servoJ_ahead_time_ = 0.0;

    std::unique_ptr<ur_rtde::RTDEControlInterface> rtde_control_;
    std::unique_ptr<ur_rtde::RTDEReceiveInterface> rtde_receive_;

    Eigen::VectorXd last_end_effector_pose_;
    Eigen::VectorXd last_arm_joint_position_;
    std::chrono::system_clock::time_point last_time_;
};

extern "C" std::unique_ptr<HardwareArm> createUR5Real() {
    return std::make_unique<UR5Real>();
}