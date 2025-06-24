/*!
* \file hardware.h
* \brief Unified Abstract Interface for all the hardware
* \author Zijian WU 
* \date 10/23/2024
*/

#ifndef HARDWARE_HPP
#define HARDWARE_HPP

#include "hardwareArm.hpp"
#include "hardwareHand.hpp"
#include "hardwareKinematic.hpp"

/* cpp library */
#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <functional>
#include <time.h>
#include <stack>

extern "C" std::unique_ptr<HardwareKinematic> createSimFK();
#ifdef BUILD_PINOCCHIO
extern "C" std::unique_ptr<HardwareKinematic> createPinocchio();
#endif

extern "C" std::unique_ptr<HardwareArm> createFlyingSim();
extern "C" std::unique_ptr<HardwareArm> createUR5Sim();
#ifdef BUILD_UR5_REAL
extern "C" std::unique_ptr<HardwareArm> createUR5Real();
#endif

extern "C" std::unique_ptr<HardwareHand> createAllegroSim();
extern "C" std::unique_ptr<HardwareHand> createLeapSim();
#ifdef BUILD_ALLEGRO_REAL
extern "C" std::unique_ptr<HardwareHand> createAllegroReal();
#endif
#ifdef BUILD_LEAP_REAL
extern "C" std::unique_ptr<HardwareHand> createLeapReal();
#endif

class Hardware
{

public:
    /**
     * construct function of the class. init all the hardware
     * @param[in] rsc_pth raisim rsc path
     * @param[in] cfg the configuration "environment" - "hardware" in yaml file 
     * @param[in] world created by raisim in environment.hpp
     */
    explicit Hardware(const std::string &rsc_pth, const Yaml::Node &cfg, std::unique_ptr<raisim::World> &world) {
        real_world_mode_ = cfg["real_world_mode"].As<bool>();
        if (!cfg["randomize_gains_hand_p"].IsNone()) {
            randomize_gains_hand_p_ = cfg["randomize_gains_hand_p"].As<double>();
        }
        if (!cfg["randomize_gains_hand_d"].IsNone()) {
            randomize_gains_hand_d_ = cfg["randomize_gains_hand_d"].As<double>();
        }
        if (!cfg["randomize_gains_arm_p"].IsNone()) {
            randomize_gains_arm_p_ = cfg["randomize_gains_arm_p"].As<double>();
        }
        if (!cfg["randomize_gains_arm_d"].IsNone()) {
            randomize_gains_arm_d_ = cfg["randomize_gains_arm_d"].As<double>();
        }
        if (!cfg["table_friction"].IsNone()) {
            table_friction_ = cfg["table_friction"].As<double>();
        }
        if (!cfg["randomize_friction"].IsNone()) {
            std::stringstream ss(cfg["randomize_friction"].As<std::string>());
            std::string token;
            while (std::getline(ss, token, ',')) {
                randomize_friction_.push_back(std::stod(token));
            }
        }

        srand(time(0));
        std::string rsc_pth_simplify = simplifyPath(rsc_pth);
        std::string type_suffix = real_world_mode_ ? "_real" : "_sim";

        setInstance(cfg["hand_type"].As<std::string>() + type_suffix, hand_map_, hand_);
        hand_->init(rsc_pth_simplify, cfg);

        setInstance(cfg["kinematic_type"].As<std::string>(), kinematic_map_, kinematic_);
        kinematic_->init(rsc_pth_simplify, cfg);

        setInstance(cfg["arm_type"].As<std::string>() + type_suffix, arm_map_, arm_);
        arm_->init(rsc_pth_simplify, cfg);

        raisim::CollisionGroup mask = 0;
        std::stringstream ss(cfg["vis_mask_id"].As<std::string>());
        std::string token;
        while (std::getline(ss, token, ',')) {
            mask = mask | raisim::COLLISION(std::stoi(token));
        }
        arm_hand_platform_ = world->addArticulatedSystem(
            rsc_pth_simplify + "/" + cfg["rsc_model"].As<std::string>() + "/" + cfg["sim_model"].As<std::string>() + ".urdf", "", {},
            raisim::COLLISION(std::stoi(cfg["vis_group_id"].As<std::string>())), mask);
        //arm_hand_platform_->setName("arm_hand_platform_");

        arm_->setSimPlatform(arm_hand_platform_);
        hand_->setSimPlatform(arm_hand_platform_);
        kinematic_->setSimPlatform(arm_hand_platform_);

        // set PD control mode
        arm_hand_platform_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
//        arm_hand_platform_->setComputeInverseDynamics(true);

        // check the gc and gv dim of the arm and hand and platform are the same
        hand_dim_ = hand_->getDim();
        arm_dim_ = arm_->getDim();
        platform_gc_dim_ = arm_hand_platform_->getGeneralizedCoordinateDim();
        platform_gv_dim_ = arm_hand_platform_->getDOF();
        if (hand_dim_ + arm_dim_ != platform_gc_dim_) {
            std::cout << "error gc and gv dim in arm, hand, platform !!!!!" << std::endl;
            exit(0);
        }
        //printf("*****finish init***** gc: hand(%d) + arm(%d) = platform(%d)\n", hand_dim_, arm_dim_, platform_gc_dim_);

        srand((unsigned)time(NULL));
    }

    /**
     * real world interaction: take action of all the joint in urdf
     * urdf in flying_hand  use prismatic joint for wrist xyz and revolute joint for wrist rxryrz (m and rad)
     * urdf in arm_hand     use revolute joit for all (rad)
     * @param[in] posTarget all the joint position act on the hardware of arm and hand. (arm first, and then hand)
     * @param[in] velTarget may be zero
     * @return None
     */
    void setPdTarget(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget) {
        //std::cout << "set:" << posTarget.transpose() << std::endl;
        if (!real_world_mode_) {
            arm_hand_platform_->setPdTarget(posTarget, velTarget);
            return;
        }

        arm_->setPdTarget(posTarget.head(arm_dim_), velTarget.head(arm_dim_));
        hand_->setPdTarget(posTarget.tail(hand_dim_), velTarget.tail(hand_dim_));
    }
    /**
     * same as `setPdTarget` but solve IK automaticly in simulation arm base Frame
     * @param[in] posTarget first is eef pose then hand joint position which are generated by policy in arm base frame
     * @param[in] velTarget may be zero
     * @return None
     */
    void setPdTargetArmBaseFrameAutoIK(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget) {
        Eigen::VectorXd new_target = posTarget;
        Eigen::VectorXd get_q(arm_dim_);
        //std::cout << "target EEF:" << new_target.head(6).transpose() << std::endl;
        HardwareKinematic::IK_ERRCODE ret = getIKSolve(new_target.head(6), get_q);
        //std::cout << "EEF result:" << get_q.transpose() << std::endl;
        new_target.head(arm_dim_) = get_q; // change the first arm_dim value into arm joint
        setPdTarget(new_target, velTarget);
    }

    /**
     * real world interaction: update all the joint state, hand frame position and orientation
     * need to update after step(), reset(), reset_state()
     * @return None
     */
    void updateObservation() {
        arm_->updateArmState();
        Eigen::VectorXd eef_pos = arm_->getEefPose();
        hand_->updateHandState(eef_pos);
        Eigen::VectorXd now_joint(platform_gc_dim_);
        now_joint.head(arm_dim_) = arm_->getJointPosition();
        now_joint.tail(hand_dim_) = hand_->getJointPosition();

        Eigen::VectorXd pinocchio_joint = now_joint;
        for (int i = 0; i < 4; i++) {
            pinocchio_joint[10 + i] = now_joint[18 + i];
            pinocchio_joint[14 + i] = now_joint[10 + i];
            pinocchio_joint[18 + i] = now_joint[14 + i];
        }

        kinematic_->updateURDFFK(pinocchio_joint);
//        std::cout<<arm_hand_platform_->getGeneralizedForce().e().tail(16).transpose()<<std::endl;
        if (real_world_mode_) {
            // Align the joints position in simulation and the real world
            Eigen::VectorXd now_joint_v(platform_gv_dim_);
            arm_hand_platform_->setState(now_joint, now_joint_v);
        }
    }

    /**
     * get the IK solution result
     * note: it will generate a new eef randomly if the IK soultion is timeout or has self-collision.
     * But return FAIL if it has been solved 10 times without a soultion.
     * But return SELF_COLLISION or TIMEOUT after a new eef is generated and is solved successfully
     * @param[in] eef 6D target end effort pose [x, y, z(m), rx, ry, rz(rad)]
     * @param[out] solved_q result of the IK. Dimensions are depended on the number of arm joints
     * @return HardwareKinematic::IK_ERRCODE
     */
    HardwareKinematic::IK_ERRCODE getIKSolve(const Eigen::VectorXd eef, Eigen::VectorXd &solved_q) {
        if (eef.isZero()) {
            solved_q.setZero(arm_dim_);
            std::cout << "eef is zero" << std::endl;
            return HardwareKinematic::IK_ERRCODE::IK_OK;
        }

        Eigen::VectorXd current_q = arm_->getJointPosition();

        HardwareKinematic::IK_ERRCODE ret = kinematic_->getArmIKSolve(eef, current_q, solved_q);
        if (HardwareKinematic::IK_ERRCODE::IK_OK == ret) {
            return ret;
        }
        std::cout << "IK err, will try again" << std::endl;

        Eigen::VectorXd new_eef = eef;
        for (int ik_cnt = 0; ik_cnt < 10; ik_cnt++) {
            for (int i = 0; i < arm_dim_; i++) {
                new_eef[i] = new_eef[i] + (rand()/double(RAND_MAX) - 0.5) * 0.2; // 0.1m and 0.1rad noise
            }

            HardwareKinematic::IK_ERRCODE tmp_ret = kinematic_->getArmIKSolve(new_eef, current_q, solved_q);
            if (HardwareKinematic::IK_ERRCODE::IK_OK == tmp_ret) {
                return ret; // return the error code for the first time of IK
            }
        }
        std::cout << "IK err 10 times. fail." << std::endl;

        return HardwareKinematic::IK_ERRCODE::IK_FAIL;
    }

    /**
     * get all the joint state
     * @param[out] genco joint position (rad)
     * @param[out] genvel joint velocity (rad/s)
     * @return None
     */
    void getState(Eigen::VectorXd &genco, Eigen::VectorXd &genvel) {
        genco.head(arm_dim_) = arm_->getJointPosition();
        genco.tail(hand_dim_) = hand_->getJointPosition();
        genvel.head(arm_dim_) = arm_->getJointVelocity();
        genvel.tail(hand_dim_) = hand_->getJointVelocity();
    }
    /**
     * set all the joint state but solve IK automaticly
     * because the first 6D of gc mean the arm eef pose
     * @param[in] genco joint position (rad) and the arm eef pose in the first 6D
     * @param[in] genvel joint velocity (rad/s)
     * @return None
     */
    void setStateArmBaseFrameAutoIK(const Eigen::VectorXd &genco, const Eigen::VectorXd &genvel) {
        Eigen::VectorXd new_target = genco;
        Eigen::VectorXd get_q(arm_dim_);
        HardwareKinematic::IK_ERRCODE ret = getIKSolve(genco.head(6), get_q);
        new_target.head(arm_dim_) = get_q; // change the first arm_dim value into arm joint
        setState(new_target, genvel);
    }

    /**
     * only use in simulation during the reset_init() (all set zero) and reset_user() method.
     * because the first 6D of gc mean the arm eef pose
     * @param[in] genco joint position (rad) and the arm eef pose in the first 6D
     * @param[in] genvel joint velocity (rad/s)
     * @return None
     */
    void setState(const Eigen::VectorXd &genco, const Eigen::VectorXd &genvel) {
        if (false == real_world_mode_) {
            arm_hand_platform_->setState(genco, genvel);
            return;
        }

        if (real_world_mode_) {
            hand_->setPdTarget(genco.tail(hand_dim_), genvel.tail(hand_dim_), false);
            arm_->setPdTarget(genco.head(arm_dim_), genvel.head(arm_dim_), false);
            usleep(100000);
            updateObservation();
            Eigen::VectorXd now_joint(platform_gc_dim_);
            now_joint.head(arm_dim_) = arm_->getJointPosition();
            now_joint.tail(hand_dim_) = hand_->getJointPosition();
            arm_hand_platform_->setState(now_joint, genvel);

        }
    }

    /**
     * Get the PD gains from arm and hand and set them in raisim. (hardware will be set individually)
     * @param[in] mode different mode of PD gain. will use only one group of PD gains by default
     * @return None
     */
    void setPdGains(int mode = 0)  {
        Eigen::VectorXd pgain(platform_gc_dim_), dgain(platform_gc_dim_);
        arm_->getPdgains(pgain, dgain, arm_dim_);
        hand_->getPdgains(pgain, dgain, hand_dim_);
        pgain.head(arm_dim_) += Eigen::VectorXd::Random(arm_dim_) * randomize_gains_arm_p_ * pgain.head(arm_dim_);
        dgain.head(arm_dim_) += Eigen::VectorXd::Random(arm_dim_) * randomize_gains_arm_d_ * dgain.head(arm_dim_);
        pgain.tail(hand_dim_) += Eigen::VectorXd::Random(hand_dim_) * randomize_gains_hand_p_ * pgain.tail(hand_dim_);
        dgain.tail(hand_dim_) += Eigen::VectorXd::Random(hand_dim_) * randomize_gains_hand_d_ * dgain.tail(hand_dim_);
        arm_hand_platform_->setPdGains(pgain, dgain);
    }

    /**
     * Get the arm base position and orientation
     * @param[in] user_input input the state by user
     * @param[in] flying_hand_mode use the state input by user if true. others may use default
     * @param[out] pos get the base position from user_input or default
     * @param[out] ori get the base orientation from user_input or default
     * @return None
     */
    void getArmBasePose(const Eigen::VectorXd &user_input, const bool flying_hand_mode, raisim::Vec<3> &pos, raisim::Mat<3,3> &ori) {
        Eigen::VectorXd pose(6);
        if (flying_hand_mode) {
            pose = user_input.head(6);
        } else {
            pose = arm_->getSimBasePose();
        }
        pos.e() = pose.head(3);
        raisim::Vec<4> quat;
        raisim::eulerToQuat(pose.segment(3,3), quat);
        raisim::quatToRotMat(quat, ori);
    }

    /**
     * Get the name and number of the bodies which are used to check contact
     * @param[out] get_vec get the name of all the bodies
     * @param[in] arm_flag choose to return the bodies of arm or hand
     * @param[in] contact_flag choose to return the bodies to check contact or get pose
     * @return number of contacts
     */
    int getBodies(std::vector<std::string> & get_vec, bool arm_flag, bool contact_flag) {
        if (arm_flag) {
            return arm_->getBodies(get_vec, contact_flag);
        } else {
            return hand_->getBodies(get_vec, contact_flag);
        }
    }

    /**
     * Get the number of fingers
     * @return number of fingers
     */
    int getNumFinger() {
        return hand_->getNumFinger();
    }

    /**
     * Get orientation in sim world frame or real arm base frame [only used for wrist]
     * @param[in] frameName the name of the frame
     * @param[out] orientation_W the rotation of the frame expressed in the world frame in raisim or armbase frame in realworld
     */
    void getFrameOrientation(const std::string &frameName, raisim::Mat<3, 3> &orientation_W) {
        kinematic_->getFrameOrientation(frameName, hand_->changeJointToLinkName(frameName), orientation_W);
    }

    /**
     * Get position in sim world frame or real arm base frame
     * @param[in] frameName the name of the frame
     * @param[out] orientation_W the position of the frame expressed in the world frame in raisim or armbase frame in realworld
     */
    void getFramePosition(const std::string &frameName, raisim::Vec<3> &point_W) {
        kinematic_->getFramePosition(frameName, hand_->changeJointToLinkName(frameName), point_W);
    }

    /**
     * Get angular velocity in sim world frame or real arm base frame [only used while training. so more accurate is better]
     * @param[in] frameName the name of the frame
     * @param[out] orientation_W the angular velocity of the frame expressed in the world frame in raisim or armbase frame in realworld
     */
    void getFrameAngularVelocity(const std::string &frameName, raisim::Vec<3> &angVel_W) {
        kinematic_->getFrameAngularVelocity(frameName, angVel_W);
    }

    /**
     * Get linear velocity in sim world frame or real arm base frame [only used while training. so more accurate is better]
     * @param[in] frameName the name of the frame
     * @param[out] vel_W the linear velocity of the frame expressed in the world frame in raisim or armbase frame in realworld
     */
    void getFrameVelocity(const std::string &frameName, raisim::Vec<3> &vel_W) {
        kinematic_->getFrameVelocity(frameName, vel_W);
    }

    /**
     * Gets the total number of controllable joints in this setting
     */
    int getGeneralizedCoordinateDim() {
        return platform_gc_dim_;
    }

    /**
     * Gets the total number of controllable joints in this setting
     */
    int getDOF() {
        return platform_gv_dim_;
    }

    /**
     * Gets the configuration of joint limits from urdf
     */
    const std::vector<raisim::Vec<2>> &getJointLimits() {
        return arm_hand_platform_->getJointLimits();
    }

    /////////////////////////////////////////////////////////////////////////////////

    // raisim API only used in raisim environment
    void setName(const std::string &name) {
        arm_hand_platform_->setName(name);
    }
    void setControlMode(raisim::ControlMode::Type mode) {
        arm_hand_platform_->setControlMode(mode);
    }
    size_t getBodyIdx(const std::string &nm) {
        return arm_hand_platform_->getBodyIdx(nm);
    }
    void setGeneralizedForce(const Eigen::VectorXd &tau) {
        arm_hand_platform_->setGeneralizedForce(tau);
    }
    void setGeneralizedCoordinate(const Eigen::VectorXd &jointState) {
        arm_hand_platform_->setGeneralizedCoordinate(jointState);
    }
    void setBasePos(const raisim::Vec<3> &pos) {
        arm_hand_platform_->setBasePos(pos);
    }
    void setBaseOrientation(const raisim::Mat<3, 3> &rot) {
        arm_hand_platform_->setBaseOrientation(rot);
    }
    std::vector<raisim::Contact> &getContacts() {
        return arm_hand_platform_->getContacts();
    }
    double getTotalMass() const {
        return arm_hand_platform_->getTotalMass();
    }
    void setMaterialFriction(std::unique_ptr<raisim::World> &world, raisim::ArticulatedSystem *arctic) {
        arctic->getCollisionBody("top/0").setMaterial("object");
        if (randomize_friction_.size() > 0) {
            double random_friction = randomize_friction_[std::rand() % randomize_friction_.size()];
            world->setMaterialPairProp("object", "object", random_friction+0.1, 0.0, 0.0);
            world->setMaterialPairProp("object", "finger", random_friction, 0.0, 0.0);
            world->setMaterialPairProp("finger", "finger", random_friction+0.1, 0.0, 0.0);
        }
        if (table_friction_ > 0.0) {
            world->setMaterialPairProp("table", "object", table_friction_, 0.0, 0.0);
        }
    }

private:
    template <typename T>
    void setInstance(const std::string &type, std::unordered_map<std::string, std::function<std::unique_ptr<T>()>> &factory, std::unique_ptr<T> &component) {
        auto it = factory.find(type);
        if (it != factory.end()) {
            component = factory[type]();
        } else {
            std::cout << "Key not found. use the default key: " << factory.begin()->first << std::endl;
            component = factory[factory.begin()->first]();
        }
    }

    std::string simplifyPath(const std::string& path) {
        std::stack<std::string> dirs;
        std::stringstream ss(path);
        std::string part;

        while (getline(ss, part, '/')) {    // segment the path
            if (part == "" || part == ".") {
                continue;
            } else if (part == "..") {
                if (!dirs.empty()) {
                    dirs.pop();
                }
            } else {
                dirs.push(part);
            }
        }

        std::string result;
        while (!dirs.empty()) {
            result = "/" + dirs.top() + result;
            dirs.pop();
        }

        return result.empty() ? "/" : result;
    }

private:
    std::unique_ptr<HardwareArm> arm_;
    std::unique_ptr<HardwareHand> hand_;
    std::unique_ptr<HardwareKinematic> kinematic_;

    raisim::ArticulatedSystem *arm_hand_platform_;

    int hand_dim_ = 0;
    int arm_dim_ = 0;
    int platform_gc_dim_ = 0;
    int hand_gv_dim_ = 0;
    int arm_gv_dim_ = 0;
    int platform_gv_dim_ = 0;

    double randomize_gains_hand_p_ = 0.0;
    double randomize_gains_hand_d_ = 0.0;
    double randomize_gains_arm_p_ = 0.0;
    double randomize_gains_arm_d_ = 0.0;
    double table_friction_ = 0.8;
    std::vector<double> randomize_friction_;

    bool real_world_mode_ = false;
    std::ofstream csv_file_;
    Eigen::VectorXd save_target_;
    Eigen::VectorXd save_current_;

    std::unordered_map<std::string, std::function<std::unique_ptr<HardwareArm>()>> arm_map_ = {
        {"ur5_sim", [](){ return createUR5Sim(); }},
        #ifdef BUILD_UR5_REAL
        {"ur5_real", [](){ return createUR5Real(); }},
        #endif
        {"flying_sim", [](){ return createFlyingSim(); }},
    };

    std::unordered_map<std::string, std::function<std::unique_ptr<HardwareHand>()>> hand_map_ = {
        {"allegro_sim", [](){ return createAllegroSim(); }},
        #ifdef BUILD_ALLEGRO_REAL
        {"allegro_real", [](){ return createAllegroReal(); }},
        #endif
        {"leap_sim", [](){ return createLeapSim(); }},
        #ifdef BUILD_LEAP_REAL
        {"leap_real", [](){ return createLeapReal(); }},
        #endif
    };

    std::unordered_map<std::string, std::function<std::unique_ptr<HardwareKinematic>()>> kinematic_map_ = {
        {"simfk", [](){ return createSimFK(); }},
        #ifdef BUILD_PINOCCHIO
        {"pinocchio", [](){ return createPinocchio(); }},
        #endif
    };
};

#endif // HARDWARE_HPP