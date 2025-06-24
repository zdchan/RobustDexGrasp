//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include "../../hardware/hardware.hpp"

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include "raisim/World.hpp"
#include <vector>
#include "raisim/math.hpp"
#include <math.h>

namespace raisim {
    class ENVIRONMENT : public RaisimGymEnv {

    public:

        explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {
            real_ = cfg["hardware"]["real_world_mode"].As<bool>();
            visualizable_ = cfg["visualize"].As<bool>();
            load_set = cfg["load_set"].As<std::string>();
            if (visualizable_) {
                std::cout<<"visualizable_: "<<visualizable_<<std::endl;
            }
            lift = false;
            lift_num = 0;

            /// create world
            world_ = std::make_unique<raisim::World>();
            world_->addGround();
            world_->setERP(0.0);

            world_->setMaterialPairProp("object", "object", 0.8, 0.0, 0.0, 0.8, 0.1);
            world_->setMaterialPairProp("object", "finger", 0.8, 0.0, 0.0, 0.8, 0.1);
            world_->setMaterialPairProp("finger", "finger", 0.8, 0.0, 0.0, 0.8, 0.1);
            world_->setDefaultMaterial(0.8, 0, 0, 0.8, 0.1);

            /// add mano
            std::string hand_model_r =  cfg["sim_model"].As<std::string>();
            if(visualizable_){
                std::cout<<"hand_model_r: "<<hand_model_r<<std::endl;
            }
            resourceDir_ = resourceDir;
            mano_r_ = std::make_unique<Hardware>(resourceDir, cfg["hardware"], world_);
            mano_r_->setName("Allegro");
//            hand_mass = mano_r_->getTotalMass();

            num_contacts = mano_r_->getBodies(contact_bodies_r_, false, true);
            num_bodyparts = mano_r_->getBodies(body_parts_r_, false, false);
            mano_r_->getBodies(contact_arm_bodies, true, true);
            mano_r_->getBodies(arm_parts, true, false);

            /// add table
            box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.771, 100, "table", raisim::COLLISION(1)));
            box->setPosition(0.2, -0.75152, 0.3855);
            box->setAppearance("0.0 0.0 0.0 0.0");

            /// set PD control mode
            mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);


            raisim::Vec<4> quat_base;
            raisim::Vec<3> euler_base;
            euler_base[0] = 0;
            euler_base[1] = 0;
            euler_base[2] = 0;
            // euler_ur5base[2] = 1.57;
            raisim::eulerToQuat(euler_base,quat_base);
            raisim::quatToRotMat(quat_base,base_mat);

            base_pos[0] = 0.;
            base_pos[1] = 0.;
            base_pos[2] = -0.;
            mano_r_->setBasePos(base_pos);

           hand_center.setZero();
           hand_center[0] = 0.08; // *2  0.107592      *1  0.0924603
           hand_center[1] = 0.01; //    -0.000996807       0.00117149
           hand_center[2] = 0.095; //    0.08785           0.10541


            /// get actuation dimensions
            gcDim_ = mano_r_->getGeneralizedCoordinateDim();
            gvDim_ = mano_r_->getDOF();

//            std::cout<<gcDim_<<std::endl;

            gc_r_.setZero(gcDim_);
            gv_r_.setZero(gvDim_);
            gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);

            /// initialize all variables
            pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_);
            actionDim_ = gcDim_;
            actionMean_r_.setZero(actionDim_);
            actionStd_r_.setOnes(actionDim_);
            joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);
            arm_gc_lift.setZero(6);

            right_hand_torque.setZero(gcDim_);

            init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
            init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
            wrist_euler_init.setZero();
            wrist_mat_r_init.setZero();
            wrist_euler_previous.setZero();
            frame_y_in_obj.setZero(num_bodyparts*3);
            joint_pos_in_obj.setZero(num_bodyparts*3);
            joint_pos_in_world.setZero(num_bodyparts*3);
            arm_joint_pos_in_world.setZero(6*3);

            joint_height_w.setZero(num_bodyparts);
            arm_height_w.setZero(6);

            init_arcticCoord.setZero(8);

            contact_body_idx_r_.setZero(num_contacts);
            contact_arm_idx.setZero(6);
            target_center_dif.setZero();

            contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
            impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
            impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);

            contacts_r_table.setZero(num_contacts); impulses_r_table.setZero(num_contacts);
            contacts_arm_table.setZero(6); impulses_arm_table.setZero(6);
            contacts_arm_all.setZero(6);

            pTarget_clipped_r.setZero(gcDim_);

            /// initialize 3D positions weights for fingertips higher than for other fingerparts
            finger_weights_contact.setOnes(num_contacts);
            for(int i=1; i < 5; i++){
                finger_weights_contact(3*i) *= 3;
            }
            finger_weights_contact.segment(10,3) *= 2;
            finger_weights_contact(num_contacts-1) *= 2;
            finger_weights_contact /= finger_weights_contact.sum();
            finger_weights_contact *= num_contacts;

            for(int i=0; i< (num_contacts - 3); i++){
                impulse_high[i] = 0.1;
                impulse_low[i] = -0.0;
            }
            for(int i=(num_contacts - 3); i< num_contacts; i++){
                impulse_high[i] = 0.2;
                impulse_low[i] = -0.0;
            }


            /// set PD gains
            mano_r_->setPdGains(0);
            mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
            mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_single = 102;
            obDim_l_ = 1;
            gsDim_ = 179;
            history_len = 10;
            tobeEncode_dim = 44;
            obDim_r_ = history_len * tobeEncode_dim + obDim_single;
            obDouble_r_.setZero(obDim_single);
            obDouble_l_.setZero(obDim_l_);
            global_state_.setZero(gsDim_);
            ob_delay_r.setZero(obDim_single);
            ob_concat_r.setZero(obDim_r_);

            float finger_action_std = cfg["finger_action_std"].As<float>();
            float rot_action_std = cfg["rot_action_std"].As<float>();

            /// retrieve joint limits from model
            joint_limits_ = mano_r_->getJointLimits();

            for(int i=0; i < int(gcDim_); i++){
                actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                joint_limit_low[i] = joint_limits_[i][0];
                joint_limit_high[i] = joint_limits_[i][1];
            }

            /// set actuation parameters
            actionStd_r_.setConstant(finger_action_std);
            actionStd_r_.head(6).setConstant(rot_action_std);
//            actionStd_r_.segment(3,3).setConstant(0.005);

            /// Initialize reward
            rewards_r_.initializeFromConfigurationFile(cfg["reward"]);

            for(int i = 0; i < num_contacts ;i++){
                contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
            }

            for(int i = 0; i < 6 ;i++){
                contact_arm_idx[i] =  mano_r_->getBodyIdx(contact_arm_bodies[i]);
                contactMapping_arm_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_arm_bodies[i])),i));
            }

            /// start visualization server
            if (visualizable_) {
                if(server_) server_->lockVisualizationServerMutex();
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();

                /// Create table
                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                table_top->setPosition(0.2, -0.75152, 0.746);
                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.746, 0.0, 0.0, 0.0, 1, "");
                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.746, 0.0, 0.0, 0.0, 1, "");
                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.746, 0.0, 0.0, 0.0, 1, "");
                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.746, 0.0, 0.0, 0.0, 1, "");
                leg1->setPosition(-0.7875,-0.28402,0.373);
                leg2->setPosition(1.1775,-0.26402,0.373);
                leg3->setPosition(-0.7875,-1.21902,0.373);
                leg4->setPosition(1.1775,-1.23902,0.373);

                obj_pose_sphere = server_->addVisualSphere("obj_pose", 0.01, 1, 0, 1, 1);
                /// initialize Cylinders for sensor
                for(int i = 0; i < num_bodyparts; i++){
                    Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                }
                for(int i = 0; i < 5; i++){
                    aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                }
                aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);
                for (int i = 0; i <200; i++) {
                    sample_point[i] = server_->addVisualSphere(std::to_string(i), 0.003, 0, 1, 0, 1);
                }

                if(server_) server_->unlockVisualizationServerMutex();
            }
        }

        void init() final { }
        void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
        /// This function loads the object into the environment
        void load_articulated(const std::string& obj_model){
            std::cout << "load obj model name is " << obj_model << std::endl;
            if(dummy_obj_flag_) {
                std::cout << "is a dummy object !!! do not show it!! " << std::endl;
                return;
            }
            arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
            arctic->setName("object");
            if(visualizable_){
                std::cout<<"obj name: "<<obj_model<<std::endl;
            }
            gcDim_obj = arctic->getGeneralizedCoordinateDim();
            gvDim_obj = arctic->getDOF();

            Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
            arctic->setGeneralizedCoordinate(gen_coord);
            arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));

            Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
            objPgain.setZero();
            objDgain.setZero();
            arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            arctic->setPdGains(objPgain, objDgain);
            arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

        }

        void set_sample_point_visual(const Eigen::Ref<EigenVec>& joint_vector) final {
            for(int i = 0; i < 200; i++) {
                raisim::Vec<3> sample_point_pos;
                sample_point_pos = joint_vector.segment(i*3,3).cast<double>();
                if (visualizable_){
                    sample_point[i]->setPosition(sample_point_pos.e());
                }
            }
        }

        bool check_collision(const Eigen::Ref<EigenVec>& joint_state) final {
            Eigen::VectorXd sc, sv;
            sc = joint_state.cast<double>();
            sv.setZero(gvDim_);
            mano_r_->setState(sc, sv);

            for (int i = 0; i < 4; i++) {
                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();
                if(server_) server_->unlockVisualizationServerMutex();
            }

            contacts_arm_all.setZero(6);

            for(auto& contact_arm: mano_r_->getContacts()) {
                contacts_arm_all[contactMapping_arm_[contact_arm.getlocalBodyIndex()]] = 1;
            }

            if (contacts_arm_all[0] > 0 || contacts_arm_all[1] > 0 || contacts_arm_all[2] > 0 || contacts_arm_all[3] > 0) {
                //std::cout << "the joint config have a self-collision" << contacts_arm_all.transpose() << std::endl;
                return false;
            } else {
                return true;
            }
        }

        void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {

            raisim::Vec<3> joint_pos_w, mesh_pos_o, mesh_pos_w, vis_cylinder_pos_w;

            for(int i = 0; i < num_bodyparts; i++) {

                mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);

                // get the point position in wrold frame
                mesh_pos_w = joint_sensor_visual.segment(i*3,3).cast<double>();

                // Given the starting and ending 3D points,
                // find the center point, rotation, and length of the visualized cylinder.
                vis_cylinder_pos_w = (joint_pos_w + mesh_pos_w) / 2.0;
                double dx = joint_pos_w[0] - mesh_pos_w[0];
                double dy = joint_pos_w[1] - mesh_pos_w[1];
                double dz = joint_pos_w[2] - mesh_pos_w[2];
                double dis = sqrt(dx*dx+dy*dy+dz*dz);

                Eigen::Vector3d A(dx,dy,dz);
                A.normalize();
                Eigen::Vector3d Z(0, 0, 1);

                // Calculate the axis of rotation
                Eigen::Vector3d axis = Z.cross(A);
                // Parallel ones require special treatment
                if (axis.norm() < 1e-6) {
                    std::cout << "In parallel" << std::endl;
                    // directions are the same, no rotation is needed
                    if (A.dot(Z) > 0) {
                        std::cout << "No rotation needed." << std::endl;
                    } else {
                        // A and Z are in opposite directions and rotated 180 degrees around the x-axis
                        axis = Eigen::Vector3d(1, 0, 0);
                    }
                }

                double angle = acos(Z.dot(A));
                Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(Z, A);

                Cylinder[i]->setPosition(vis_cylinder_pos_w.e());
                Cylinder[i]->setOrientation(q.w(), q.x(), q.y(), q.z());
                Cylinder[i]->setCylinderSize(0.001, dis);
            }

        }

        /// Resets the object and hand to its initial pose
        void reset() final {
            if (first_reset_)
            {
                first_reset_=false;
            }
            else{
                lift = false;
                lift_num = 0;
//                std::cout<<"reset!!!!!!!!!!!!!!!!!!!!"<<std::endl;
                /// all settings to initial state configuration
                actionMean_r_.setZero();
                mano_r_->setBasePos(base_pos);
                mano_r_->setBaseOrientation(base_mat);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                box->clearExternalForcesAndTorques();
                box->setPosition(0.2, -0.75152, 0.3855);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                gc_r_=gc_set_r_;
                right_hand_torque.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gv_set_r_.setZero(gvDim_);
                pTarget_r_ = gc_set_r_;
                pTarget_clipped_r = gc_set_r_;
                vTarget_r_.setZero(gvDim_);
                actionMean_r_.setZero();
//                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);
                actionMean_r_ = gc_set_r_;
                wrist_mat_r_init.setZero();
                wrist_euler_previous.setZero();
                updateObservation();
            }
        }
        /// Resets the state to a user defined input
        // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
        // init_state_l in right-hand coord
        void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                         const Eigen::Ref<EigenVec>& init_state_l,
                         const Eigen::Ref<EigenVec>& init_vel_r,
                         const Eigen::Ref<EigenVec>& init_vel_l,
                         const Eigen::Ref<EigenVec>& obj_pose) final {
            lift = false;
            lift_num = 0;
            obs_history.clear();
            /// reset gains (only required in case for inference)
            mano_r_->setPdGains(0);

            Eigen::VectorXd gen_force;
            gen_force.setZero(gcDim_);
            mano_r_->setGeneralizedForce(gen_force);

            /// reset table position (only required in case for inference)
            box->setPosition(0.2, -0.75152, 0.3855);
            box->setOrientation(1,0,0,0);
            box->setVelocity(0,0,0,0,0,0);

            mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));

            gc_set_r_ = init_state_r.cast<double>(); //.cast<double>();
            gv_set_r_ = init_vel_r.cast<double>(); //.cast<double>();

            /// set initial root position in global frame as origin in new coordinate frame
//            init_root_r_  = init_state_r.head(3);
            init_obj_ = obj_pose.head(3).cast<double>();
            if (visualizable_){
                obj_pose_sphere->setPosition(init_obj_.e());
            }

            /// set initial root orientation in global frame as origin in new coordinate frame
            raisim::Vec<4> quat;
            raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
            raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
            raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse

            if (dummy_obj_flag_ == false) {
                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>().tail(arcticCoordDim);

                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
            }

            mano_r_->setBasePos(base_pos);
            mano_r_->setBaseOrientation(base_mat);
            mano_r_->setState(gc_set_r_, gv_set_r_);

            actionMean_r_ = gc_set_r_;

            gen_force.setZero(gcDim_);
            mano_r_->setGeneralizedForce(gen_force);

            mano_r_->updateObservation();
            mano_r_->getState(gc_r_, gv_r_);
            pTarget_clipped_r = gc_r_;
            right_hand_torque.setZero(gcDim_);

            raisim::Mat<3,3> wrist_mat_r;
            mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
            raisim::RotmatToEuler(wrist_mat_r, wrist_euler_init);
            wrist_mat_r_init = wrist_mat_r;
			wrist_euler_previous.setZero();
            updateObservation();
        }

        void update_target(const Eigen::Ref<EigenVec>& target_center) final {
        }

        void set_goals(const Eigen::Ref<EigenVec>& obj_angle,
                       const Eigen::Ref<EigenVec>& obj_pos,
                       const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                       const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                       const Eigen::Ref<EigenVec>& goal_pose_r,
                       const Eigen::Ref<EigenVec>& goal_pose_l,
                       const Eigen::Ref<EigenVec>& goal_qpos_r,
                       const Eigen::Ref<EigenVec>& goal_qpos_l,
                       const Eigen::Ref<EigenVec>& goal_contacts_r,
                       const Eigen::Ref<EigenVec>& goal_contacts_l) final {}

        /// This function takes an environment step given an action (26DoF) input
        // action_l in left-hand coord
        float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {
            /// Compute position target for actuators
            pTarget_r_ = action_r.cast<double>();
            pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
            pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)
            if (lift){
                lift_num += 1;
                if(lift_num > 80) lift_num = 80;
                pTarget_r_.head(6) = arm_gc_lift + (action_r.cast<double>().head(6) - arm_gc_lift) * lift_num / 80;
            }

            /// Clip targets to limits
            pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

            /// Apply N control steps
            double delay_cnt = 1.0;

            Eigen::VectorXd vec = pTarget_clipped_r.head(6) - gc_r_.head(6);

            double max_step_distance_arm = 0.0, max_step_distance_hand = 0.0;
            double arm_delay_cnt = 1.0, hand_delay_cnt = 1.0;

            max_step_distance_arm = abs(vec(1));
            if (max_step_distance_arm < abs(vec(2))) {
                max_step_distance_arm = abs(vec(2));
            }
            if (max_step_distance_arm > 0.03) {
                arm_delay_cnt = round(max_step_distance_arm / 0.025) + 1.0;
                if (arm_delay_cnt > 5.0) {
                    arm_delay_cnt = 5.0;
                }
            }

            if (arm_delay_cnt > hand_delay_cnt) delay_cnt = arm_delay_cnt;
            else delay_cnt = hand_delay_cnt;

            //std::cout << "target pose = " << pTarget_clipped_r.transpose() << std::endl;
            Eigen::VectorXd tmp_gc_r_ = gc_r_;
            for (int step = 1; step <= int(delay_cnt); step++) {
                double step_distance = 0.0;
                Eigen::VectorXd pTarget_clipped_step = pTarget_clipped_r;
                for (int i = 0; i < 6; i++) {
                    pTarget_clipped_step[i] = gc_r_[i] + (pTarget_clipped_r[i] - gc_r_[i]) / delay_cnt * step;
                }

                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_step, vTarget_r_);

                /// Apply N control steps
                int step_cnt = 0;
                auto starttime = std::chrono::system_clock::now();

                while (1) {
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                    auto diff_time = std::chrono::system_clock::now() - starttime;
                    if (diff_time.count() / 1e9 > control_dt_) {
                        break;
                    }
                }

                mano_r_->updateObservation();
                mano_r_->getState(gc_r_, gv_r_);
            }

            updateObservation();

            actionMean_r_ = gc_r_;
            rewards_sum_[0] = 0;
            rewards_sum_[1] = 0;
            return rewards_sum_;
        }


        /// This function computes and updates the observation/state space
        void updateObservation() {
            mano_r_->updateObservation();

            raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
            mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
            raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
            mano_r_->getState(gc_r_, gv_r_);

            
            right_hand_torque = (pTarget_clipped_r - gc_r_);

            raisim::Vec<3> wrist_pos_w;
            mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

            Eigen::Vector3d hand_center_w;
            hand_center_w = wrist_mat_r.e() * hand_center;
            hand_center_w[0] += wrist_pos_w[0];
            hand_center_w[1] += wrist_pos_w[1];
            hand_center_w[2] += wrist_pos_w[2];

            raisim::Vec<3> joint_pos_w;
            for(int i = 0; i < num_bodyparts ; i++){
                mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                joint_pos_in_world[i * 3] = joint_pos_w[0];
                joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];
                joint_height_w[i] = joint_pos_w[2] - 0.771;
            }

            for(int i = 0; i < 6 ; i++){
                mano_r_->getFramePosition(arm_parts[i], joint_pos_w);
                arm_joint_pos_in_world[i * 3] = joint_pos_w[0];
                arm_joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                arm_joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];
                arm_height_w[i] = joint_pos_w[2] - 0.771;
            }

            raisim::Vec<3> wrist_euler_current;
            raisim::RotmatToEuler(wrist_mat_r, wrist_euler_current);

            if (wrist_euler_previous.norm() > 0.01){
                for (int i = 0; i < 3; i++) {
                    if (wrist_euler_current[i] - wrist_euler_previous[i] > M_PI) {
                        wrist_euler_current[i] -= 2 * M_PI;
                } else if (wrist_euler_current[i] - wrist_euler_previous[i] < -M_PI) {
                        wrist_euler_current[i] += 2 * M_PI;
                    }
                }
            }

            wrist_euler_previous = wrist_euler_current;

            Eigen::Vector3d euler_diff;

            raisim::Vec<3> euler_diff_raisim;
            raisim::Mat<3,3> wrist_mat_r_init_trans, wrist_mat_diff;
            raisim::transpose(wrist_mat_r_init, wrist_mat_r_init_trans);
            raisim::matmul(wrist_mat_r_init_trans, wrist_mat_r, wrist_mat_diff);
            raisim::RotmatToEuler(wrist_mat_diff, euler_diff_raisim);
            euler_diff = euler_diff_raisim.e();
//            euler_diff = wrist_euler_current.e() - wrist_euler_init.e();

                obDouble_r_ << gc_r_,
                            right_hand_torque,
                            contacts_r_af,
                            impulses_r_af,
                            joint_height_w,
                            arm_height_w,
                            hand_center_w,
                            euler_diff,
                            // 0,0,0;
							wrist_euler_current.e();
            obs_history.push_back(obDouble_r_);

           raisim::Vec<3> obj_pose, wrist_pos_obj, hand_pose_trans, obj_pose_wrist;
           obj_pose.setZero();wrist_pos_obj.setZero();obj_pose_wrist.setZero();Obj_Position.setZero();
           raisim::RotmatToEuler(wrist_mat_r_trans, hand_pose_trans);

            global_state_ << obj_pose_wrist.e(),
                             frame_y_in_obj,
                             joint_pos_in_obj,
                             Obj_Position.e(),
                             hand_pose_trans.e(),
                             0,
                             wrist_pos_w.e(),
                             target_center_dif,
                             obj_pose.e(),
                             wrist_pos_obj.e(),
                             contacts_arm_all[1],
                             contacts_arm_all[2],
                             contacts_arm_all[3],
                             contacts_arm_all[4],
                             joint_pos_in_world;
        }

        /// Set observation in wrapper to current observation
        void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
            int lag = 1;
            int vec_size = obs_history.size();
            ob_delay_r << obs_history[vec_size - lag];
            if (vec_size == 1)  //
            {
                for (int i = 1; i < history_len; i++)
                {
                    obs_history.push_back(obDouble_r_);
                }
            }
            vec_size = obs_history.size();
            for (int i = 0; i < history_len; i++)
            {
                ob_concat_r.segment(tobeEncode_dim * i, tobeEncode_dim) << obs_history[vec_size - history_len + i].head(tobeEncode_dim);
            }
            ob_concat_r.tail(obDim_single) << ob_delay_r;

            ob_r = ob_concat_r.cast<float>();
            ob_l = obDouble_l_.cast<float>();
        }

        void get_global_state(Eigen::Ref<EigenVec> gs) {
            gs = global_state_.cast<float>();
        }


        void set_rootguidance() final {}
        void switch_root_guidance(bool is_on) {
            lift = true;
            arm_gc_lift = gc_r_.head(6);
            lift_num = 0;
        }
        /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
        bool isTerminalState(float& terminalReward) final {
            for(int i = 0; i < num_bodyparts ; i++){
                if (joint_height_w[i] < -0.008){
                    terminalReward = -10;
                    std::cout<<"joint_height_w: "<<joint_height_w[i]<<std::endl;
                    return true;
                }
            }

            if(obDouble_r_.hasNaN() || global_state_.hasNaN())
            {
                std::cout<<"NaN detected"<< obDouble_r_.transpose()<<std::endl<<std::endl<<std::endl;
                std::cout<<"NaN detected"<< global_state_.transpose()<<std::endl<<std::endl<<std::endl;
                return true;
            }

            return false;
        }

    private:
        int gcDim_, gvDim_, tobeEncode_dim, history_len, obDim_single;
        int gcDim_obj, gvDim_obj;
        bool real_ = false;
        bool visualizable_ = false;
        bool lift = false;
        bool dummy_obj_flag_ = true;

        int lift_num = 0;
        raisim::ArticulatedSystem* mano_;
        Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, vTarget_r_, gc_set_r_, gv_set_r_;
        Eigen::VectorXd joint_pos_in_world;
        Eigen::VectorXd arm_joint_pos_in_world;
        std::string load_set;

        int num_contacts = 0;
        int num_bodyparts = 0;

        raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_;
        raisim::Vec<3> init_root_r_, init_obj_;
        Eigen::VectorXd joint_limit_high, joint_limit_low;
        Eigen::VectorXd impulse_high, impulse_low;
        Eigen::VectorXd actionMean_r_, actionStd_r_, arm_gc_lift;
        Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_, ob_delay_r, ob_concat_r;
        Eigen::VectorXd finger_weights_contact, finger_weights_aff;
        Eigen::VectorXd contacts_r_af, impulses_r_af;
        Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
        Eigen::VectorXd contacts_r_table, impulses_r_table;
        Eigen::VectorXd contacts_arm_table, impulses_arm_table, contacts_arm_all;
        Eigen::VectorXd contact_body_idx_r_, contact_arm_idx;
        Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj, joint_height_w, arm_height_w;
        Eigen::VectorXd right_hand_torque;
        Eigen::VectorXd pTarget_clipped_r, pTarget_prev_r;
        Eigen::Vector3d hand_center;
        std::deque<Eigen::VectorXd> obs_history;

        Eigen::VectorXd init_arcticCoord;

        raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
        raisim::Box *box;
        raisim::ArticulatedSystem *arctic;
        std::unique_ptr<Hardware> mano_r_; 
        raisim::ArticulatedSystemVisual *arcticVisual;
        raisim::Vec<3> wrist_euler_init;
        raisim::Vec<3> Obj_Position;
        Eigen::Vector3d target_center_dif;
        bool first_reset_=true;
        float rewards_sum_[2];
        bool has_non_aff = false;
        std::vector<std::string> body_parts_r_;
        std::vector<std::string> contact_bodies_r_;
        std::vector<std::string> arm_parts;
        std::vector<std::string> contact_arm_bodies;

        raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4;
        raisim::Visuals *Cylinder[17];
        raisim::Visuals *aff_center_visual[7];
        raisim::Visuals *wrist_target[2];
        raisim::Visuals *obj_pose_sphere;
        raisim::Visuals *sample_point[200];
        
        raisim::Vec<3> base_pos;
        raisim::Mat<3,3> base_mat;
        raisim::Mat<3,3> wrist_mat_r_init;
        raisim::Vec<3> wrist_euler_previous;

        std::map<int,int> contactMapping_arm_;
        std::string resourceDir_;
        std::vector<raisim::Vec<2>> joint_limits_;
        raisim::PolyLine *line;
    };
}
