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

            visualizable_ = cfg["visualize"].As<bool>();
            load_set = cfg["load_set"].As<std::string>();
            if (visualizable_) {
                std::cout<<"visualizable_: "<<visualizable_<<std::endl;
            }
            lift = false;
            lift_num = 0;

            // Create world
            world_ = std::make_unique<raisim::World>();
            world_->addGround();
            world_->setERP(0.0);

            world_->setMaterialPairProp("object", "object", 0.8, 0.0, 0.0, 0.8, 0.1);
            world_->setMaterialPairProp("object", "finger", 0.8, 0.0, 0.0, 0.8, 0.1);
            world_->setMaterialPairProp("finger", "finger", 0.8, 0.0, 0.0, 0.8, 0.1);
            world_->setDefaultMaterial(0.8, 0, 0, 0.8, 0.1);

            // Add hand model
            std::string hand_model_r = cfg["hardware"]["sim_model"].As<std::string>();
            if(visualizable_){
                std::cout<<"hand_model_r: "<<hand_model_r<<std::endl;
            }
            resourceDir_ = resourceDir;
            mano_r_ = std::make_unique<Hardware>(resourceDir, cfg["hardware"], world_);
            mano_r_->setName("Allegro");

            num_contacts = mano_r_->getBodies(contact_bodies_r_, false, true);
            num_bodyparts = mano_r_->getBodies(body_parts_r_, false, false);
            mano_r_->getBodies(contact_arm_bodies, true, true);
            mano_r_->getBodies(arm_parts, true, false);

            // Add table
            box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.771, 100, "table", raisim::COLLISION(1)));
            box->setPosition(0.2, -0.75152, 0.3855);
            box->setAppearance("0.0 0.0 0.0 0.0");

            // Set PD control mode
            mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

            raisim::Vec<4> quat_base;
            raisim::Vec<3> euler_base;
            euler_base[0] = 0;
            euler_base[1] = 0;
            euler_base[2] = 0;
            raisim::eulerToQuat(euler_base,quat_base);
            raisim::quatToRotMat(quat_base,base_mat);

            base_pos[0] = 0.;
            base_pos[1] = 0.;
            base_pos[2] = -0.;
            mano_r_->setBasePos(base_pos);

            /// Set hand center
            hand_center.setZero();
            std::string hand_center_str = cfg["hardware"]["hand_center"].As<std::string>();
            // remove square brackets
            hand_center_str = hand_center_str.substr(1, hand_center_str.length() - 2);
            // split string
            std::stringstream ss(hand_center_str);
            std::string item;
            std::vector<float> values;
            while (std::getline(ss, item, ',')) {
                // remove space
                item.erase(remove_if(item.begin(), item.end(), isspace), item.end());
                values.push_back(std::stof(item));
            }
            hand_center[0] = values[0];
            hand_center[1] = values[1];
            hand_center[2] = values[2];

            // Get actuation dimensions
            gcDim_ = mano_r_->getGeneralizedCoordinateDim();
            gvDim_ = mano_r_->getDOF();

            gc_r_.setZero(gcDim_);
            gv_r_.setZero(gvDim_);
            gc_set_r_.setZero(gcDim_); 
            gv_set_r_.setZero(gvDim_);

            // Initialize all variables
            pTarget_r_.setZero(gcDim_); 
            vTarget_r_.setZero(gvDim_);
            actionDim_ = gcDim_;
            actionMean_r_.setZero(actionDim_);
            actionStd_r_.setOnes(actionDim_);
            joint_limit_high.setZero(actionDim_); 
            joint_limit_low.setZero(actionDim_);
            arm_gc_lift.setZero(6);

            right_hand_torque.setZero(gcDim_);

            init_or_r_.setZero();  
            init_rot_r_.setZero(); 
            init_root_r_.setZero();
            init_obj_rot_.setZero(); 
            init_obj_or_.setZero(); 
            init_obj_.setZero();
            wrist_mat_r_in_obj_init.setZero();
            wrist_euler_in_obj_init.setZero();
            wrist_euler_init.setZero();
            wrist_mat_r_init.setZero();
            wrist_euler_previous.setZero();
            wrist_vel.setZero(); 
            wrist_qvel.setZero(); 
            wrist_vel_in_wrist.setZero(); 
            wrist_qvel_in_wrist.setZero();
            afford_center.setZero();
            obj_base_pos.setZero();
            frame_y_in_obj.setZero(num_bodyparts*3);
            joint_pos_in_obj.setZero(num_bodyparts*3);
            joint_pos_in_world.setZero(num_bodyparts*3);
            arm_joint_pos_in_world.setZero(6*3);

            joint_height_w.setZero(num_bodyparts);
            arm_height_w.setZero(6);

            init_arcticCoord.setZero(8);

            contact_body_idx_r_.setZero(num_contacts);
            contact_arm_idx.setZero(6);

            obj_pos_init_.setZero(8);
            Position.setZero();
            Obj_Position.setZero(); 
            Obj_Position_init.setZero(); 
            Obj_orientation.setZero(); 
            Obj_orientation_temp.setZero(); 
            Obj_orientation_init.setZero();
            obj_quat.setZero();
            Obj_qvel.setZero(); 
            Obj_linvel.setZero();
            obj_vel_in_wrist.setZero(); 
            obj_qvel_in_wrist.setZero();
            target_center_dif.setZero();

            contacts_r_af.setZero(num_contacts); 
            contacts_r_non_af.setZero(num_contacts);
            impulses_r_af.setZero(num_contacts); 
            impulses_r_non_af.setZero(num_contacts);
            impulses_r_af_vector.setZero(num_contacts*3); 
            impulses_r_non_af_vector.setZero(num_contacts*3);
            impulses_r_table_vector.setZero(num_contacts*3); 
            impulses_arm_table_vector.setZero(6*3);
            impulses_r_af_xy.setZero(num_contacts*2); 
            impulses_r_af_z.setZero(num_contacts);
            impulse_high.setZero(num_contacts); 
            impulse_low.setZero(num_contacts);

            contacts_r_table.setZero(num_contacts); 
            impulses_r_table.setZero(num_contacts);
            contacts_arm_table.setZero(6); 
            impulses_arm_table.setZero(6);
            contacts_arm_all.setZero(6);

            pTarget_clipped_r.setZero(gcDim_);
            pTarget_prev_r.setZero(gcDim_);

            // Initialize 3D positions weights for fingertips higher than for other fingerparts
            finger_weights_contact.setOnes(num_contacts);
            for(int i=1; i < 5; i++){
                finger_weights_contact(3*i) *= 3;
            }
            finger_weights_contact(0) = 0;
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

            // Set PD gains
            mano_r_->setPdGains(0);
            mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
            mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

            // Initialize environment dimensions
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

            // Retrieve joint limits from model
            joint_limits_ = mano_r_->getJointLimits();

            for(int i=0; i < int(gcDim_); i++){
                actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                joint_limit_low[i] = joint_limits_[i][0];
                joint_limit_high[i] = joint_limits_[i][1];
            }

            // Set actuation parameters
            actionStd_r_.setConstant(finger_action_std);
            actionStd_r_.head(6).setConstant(rot_action_std);

            // Initialize reward
            rewards_r_.initializeFromConfigurationFile(cfg["reward"]);

            for(int i = 0; i < num_contacts ;i++){
                contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
            }

            for(int i = 0; i < 6 ;i++){
                contact_arm_idx[i] =  mano_r_->getBodyIdx(contact_arm_bodies[i]);
                contactMapping_arm_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_arm_bodies[i])),i));
            }

            // Start visualization server
            if (visualizable_) {
                if(server_) server_->lockVisualizationServerMutex();
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();

                // Create table
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

                // Initialize Cylinders for sensor
                for(int i = 0; i < num_bodyparts; i++){
                    Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                    sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                    joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                }
                for(int i = 0; i < 5; i++){
                    aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                }
                aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);

                if(server_) server_->unlockVisualizationServerMutex();
            }
        }

        void init() final { }
        void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
        
        // This function loads the object into the environment
        void load_articulated(const std::string& obj_model){
            obj_name = obj_model;
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
            obj_weight = arctic->getTotalMass();

            Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
            objPgain.setZero();
            objDgain.setZero();
            arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            arctic->setPdGains(objPgain, objDgain);
            arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

            auto non_affordance_id = arctic->getBodyIdx("bottom");
            double non_aff_mass = arctic->getMass(non_affordance_id);
            has_non_aff = (non_aff_mass > 0.001);
        }

        void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {

            raisim::Vec<3> joint_pos_w, mesh_pos_o, mesh_pos_w, vis_cylinder_pos_w;

            for(int i = 0; i < num_bodyparts; i++) {

                mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);

                // Get the point position in world frame
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
                
                // All settings to initial state configuration
                actionMean_r_.setZero();
                mano_r_->setBasePos(base_pos);
                mano_r_->setBaseOrientation(base_mat);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                gvDim_obj = arctic->getDOF();
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(init_arcticCoord, Eigen::VectorXd::Zero(gvDim_obj));

                box->clearExternalForcesAndTorques();
                box->setPosition(0.2, -0.75152, 0.3855);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                auto affordance_id = arctic->getBodyIdx("top");
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                gen_force.setZero(gcDim_obj);
                arctic->setGeneralizedForce(gen_force);

                gc_r_=gc_set_r_;
                right_hand_torque.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gv_set_r_.setZero(gvDim_);
                pTarget_r_ = gc_set_r_;
                pTarget_clipped_r = gc_set_r_;
                pTarget_prev_r = gc_set_r_;
                vTarget_r_.setZero(gvDim_);
                actionMean_r_ = gc_set_r_;
                wrist_vel.setZero(); 
                wrist_qvel.setZero(); 
                wrist_vel_in_wrist.setZero(); 
                wrist_qvel_in_wrist.setZero();
                Obj_Position.setZero();
                Obj_orientation_temp.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); 
                Obj_linvel.setZero();
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
            
            // Reset gains (only required in case for inference)
            mano_r_->setPdGains(0);

            Eigen::VectorXd gen_force;
            gen_force.setZero(gcDim_);
            mano_r_->setGeneralizedForce(gen_force);

            // Reset state
            pTarget_clipped_r.setZero(gcDim_); 
            pTarget_prev_r.setZero(gcDim_);

            // Reset table position (only required in case for inference)
            box->setPosition(0.2, -0.75152, 0.3855);
            box->setOrientation(1,0,0,0);
            box->setVelocity(0,0,0,0,0,0);

            Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
            objPgain.setZero();
            objDgain.setZero();
            arctic->setPdGains(objPgain, objDgain);

            mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
            arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

            gc_set_r_ = init_state_r.cast<double>();
            gv_set_r_ = init_vel_r.cast<double>();
            mano_r_->setState(gc_set_r_, gv_set_r_);

            pTarget_clipped_r = gc_set_r_;
            pTarget_prev_r = gc_set_r_;

            // Set initial root position in global frame as origin in new coordinate frame
            init_obj_ = obj_pose.head(3).cast<double>();

            // Set initial root orientation in global frame as origin in new coordinate frame
            raisim::Vec<4> quat;
            raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
            raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
            raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse

            int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
            int arcticVelDim = arctic->getDOF();
            Eigen::VectorXd arcticCoord, arcticVel;
            arcticCoord.setZero(arcticCoordDim);
            arcticVel.setZero(arcticVelDim);
            arcticCoord = obj_pose.cast<double>().tail(arcticCoordDim);

            init_arcticCoord = arcticCoord;

            raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
            raisim::transpose(init_obj_rot_, init_obj_or_);
            arctic->setBasePos(init_obj_);
            arctic->setBaseOrientation(init_obj_rot_);
            arctic->setState(arcticCoord, arcticVel);
            mano_r_->setBasePos(base_pos);
            mano_r_->setBaseOrientation(base_mat);
            mano_r_->setState(gc_set_r_, gv_set_r_);

            // Set initial object state
            obj_pos_init_  = obj_pose.cast<double>(); // 8 dof

            // Set action mean to initial pose
            actionMean_r_ = gc_set_r_;

            gen_force.setZero(gcDim_);
            mano_r_->setGeneralizedForce(gen_force);

            obj_weight = arctic->getTotalMass();

            mano_r_->setMaterialFriction(world_, arctic);

            mano_r_->updateObservation();

           auto affordance_id = arctic->getBodyIdx("top");
           arctic->getOrientation(affordance_id, Obj_orientation_init);
           raisim::Mat<3,3> wrist_mat_r;
           mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
           raisim::Mat<3,3> Obj_orientation_init_trans;
           raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
           raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
           raisim::RotmatToEuler(wrist_mat_r_in_obj_init, wrist_euler_in_obj_init);
           raisim::RotmatToEuler(wrist_mat_r, wrist_euler_init);
           wrist_mat_r_init = wrist_mat_r;
           wrist_euler_previous.setZero();

            updateObservation();
        }

        void update_target(const Eigen::Ref<EigenVec>& target_center) final {
            afford_center = target_center.cast<double>();
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

            auto affordance_id = arctic->getBodyIdx("top");
            raisim::Vec<3> obj_pos_w, target_center, afford_center_w;
            raisim::Mat<3,3> obj_rot_w;
            arctic->getPosition(affordance_id, obj_pos_w);
            arctic->getOrientation(affordance_id, obj_rot_w);
            raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);

            raisim::Vec<3> wrist_pos_w;
            raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
            mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
            mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

            target_center[0] = afford_center_w[0] + obj_pos_w[0];
            target_center[1] = afford_center_w[1] + obj_pos_w[1];
            target_center[2] = afford_center_w[2] + obj_pos_w[2];
            if (visualizable_){
                aff_center_visual[6]->setPosition(target_center.e());
            }

            Eigen::Vector3d hand_center_w;
            hand_center_w = wrist_mat_r.e() * hand_center;
            hand_center_w[0] += wrist_pos_w[0];
            hand_center_w[1] += wrist_pos_w[1];
            hand_center_w[2] += wrist_pos_w[2];

            if (visualizable_){
                aff_center_visual[5]->setPosition(hand_center_w);
            }

            // Compute position target for actuators
            pTarget_r_ = action_r.cast<double>();
            pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); // Residual action * scaling
            pTarget_r_ += actionMean_r_; // Add current state
            if (lift){
                lift_num += 1;
                if(lift_num > 80) lift_num = 80;
                pTarget_r_.head(6) = arm_gc_lift + (action_r.cast<double>().head(6) - arm_gc_lift) * lift_num / 80;
            }

            // Clip targets to limits
            pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

            // Randomly sample delay_flag with C++:
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 1);
            bool delay_flag = dis(gen);

            // Apply N control steps
            for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                if((delay_flag == 0)&&(i == 0)){
                    mano_r_->setPdTarget(pTarget_prev_r, vTarget_r_);
                }
                else{
                    mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);
                }

                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();
                if(server_) server_->unlockVisualizationServerMutex();
            }

            pTarget_prev_r = pTarget_clipped_r;

            // Update observation and set new mean to the latest pose
            updateObservation();
            actionMean_r_ = gc_r_;

            affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
            if (has_non_aff){
                not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
            }
            else{
                not_affordance_contact_reward_r = 0;
            }
            table_contact_reward_r = contacts_r_table.cwiseProduct(finger_weights_contact).sum() / num_contacts;

            Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped, impulses_r_table_clipped;
            impulses_r_af_clipped.setZero(num_contacts);
            impulses_r_non_af_clipped.setZero(num_contacts);
            impulses_r_table_clipped.setZero(num_contacts);
            // impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
            impulses_r_af_clipped = impulses_r_af_xy.cwiseMax(impulse_low).cwiseMin(impulse_high);
            impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
            impulses_r_table_clipped = impulses_r_table.cwiseMax(impulse_low).cwiseMin(impulse_high);


            affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
            if (has_non_aff){
                not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
            }
            else{
                not_affordance_impulse_reward_r = 0;
            }

            // Calculate push reward based on sum of impulses_r_af_z
            push_reward_r = 0;
            if (impulses_r_af_z[0] > 1.0){
                push_reward_r += (impulses_r_af_z[0] - 1.0);
            }
            for (int i = 1; i < num_contacts; i++){
                if (impulses_r_af_z[i] > 2.0){
                    push_reward_r += (impulses_r_af_z[i] - 2.0);
                }
            }
            if (push_reward_r > 10.0){
                push_reward_r = 10.0;
            }
            
            table_impulse_reward_r = impulses_r_table_clipped.cwiseProduct(finger_weights_contact).sum();

            arm_table_contact_reward = contacts_arm_table.norm();
            arm_table_impulse_reward = impulses_arm_table.norm();

            obj_displacement_reward = (Obj_Position.e() - obj_pos_init_.head(3)).norm();

            wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
            wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
            obj_vel_reward_r = Obj_linvel.e().squaredNorm();
            obj_qvel_reward_r = Obj_qvel.e().squaredNorm();

            if(wrist_vel_in_wrist.norm() > 0.25){
                wrist_vel_reward_r *= 10;
            }

            Eigen::VectorXd arm_joint_vel = gv_r_.head(6);
            for(int i = 0; i < 6; i++){
                if(arm_joint_vel[i] > 0.5){
                    arm_joint_vel[i] *= 4;
                }
                if(arm_joint_vel[i] < -0.5){
                    arm_joint_vel[i] *= 4;
                }
            }
            arm_joint_vel_reward = arm_joint_vel.squaredNorm();


           raisim::Mat<3,3> obj_rot_w_trans, wrist_mat_r_in_obj;
           raisim::Vec<3> wrist_euler_in_obj;
           mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
           arctic->getOrientation(affordance_id, obj_rot_w);
           raisim::transpose(obj_rot_w, obj_rot_w_trans);
           raisim::matmul(obj_rot_w_trans, wrist_mat_r, wrist_mat_r_in_obj);
           raisim::RotmatToEuler(wrist_mat_r_in_obj, wrist_euler_in_obj);


            rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
            rewards_r_.record("push_reward", std::max(0.0, push_reward_r));
            rewards_r_.record("affordance_impulse_reward", std::max(0.0, affordance_impulse_reward_r));
            rewards_r_.record("table_contact_reward", std::max(0.0, table_contact_reward_r));
            rewards_r_.record("table_impulse_reward", std::max(0.0, table_impulse_reward_r));
            rewards_r_.record("obj_displacement_reward", std::max(0.0, obj_displacement_reward));
            rewards_r_.record("arm_contact_reward", std::max(0.0, arm_table_contact_reward));
            rewards_r_.record("arm_impulse_reward", std::max(0.0, arm_table_impulse_reward));
            rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
            rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
            rewards_r_.record("arm_joint_vel_reward_", std::max(0.0, arm_joint_vel_reward));
            rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
            rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));

            rewards_sum_[0] = rewards_r_.sum();
            rewards_sum_[1] = 0;

            return rewards_sum_;
        }

        /// This function computes and updates the observation/state space
        void updateObservation() {
            mano_r_->updateObservation();
            
            contacts_r_af.setZero();
            contacts_r_non_af.setZero();
            impulses_r_af.setZero();
            impulses_r_non_af.setZero();
            contacts_r_table.setZero();
            impulses_r_table.setZero();
            contacts_arm_table.setZero();
            impulses_arm_table.setZero();
            contacts_arm_all.setZero();
            impulses_r_af_vector.setZero();
            impulses_r_non_af_vector.setZero();
            impulses_r_table_vector.setZero();
            impulses_arm_table_vector.setZero();
            impulses_r_af_xy.setZero();
            impulses_r_af_z.setZero();

            raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
            mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
            raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
            mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
            mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);

            wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
            wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();

            mano_r_->getState(gc_r_, gv_r_);

            // Get updated object pose
            auto affordance_id = arctic->getBodyIdx("top");
            auto non_affordance_id = arctic->getBodyIdx("bottom");
            arctic->getPosition(affordance_id, Obj_Position);
            arctic->getOrientation(affordance_id, Obj_orientation_temp);
            raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
            arctic->getAngularVelocity(affordance_id, Obj_qvel);
            arctic->getVelocity(affordance_id, Obj_linvel);

            // Object velocity in wrist frame
            obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist;
            obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist;

            // Compute current contacts of hand parts and the contact force
            auto& contact_list_obj = arctic->getContacts();

            for(auto& contact_af: mano_r_->getContacts()) {
                if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                if (contact_af.getImpulse().e().norm() < 0.001) continue;
                int idx = contactMapping_r_[contact_af.getlocalBodyIndex()];
                Eigen::Vector3d impulse = contact_af.getContactFrame().e().transpose() * contact_af.getImpulse().e();
                if (!contact_af.isObjectA()) impulse = -impulse;
                impulses_r_af_vector[idx*3] += impulse[0];
                impulses_r_af_vector[idx*3+1] += impulse[1];
                impulses_r_af_vector[idx*3+2] += impulse[2];
            }

            for(int i = 0; i < num_contacts; i++){
                impulses_r_af[i] = impulses_r_af_vector.segment<3>(i*3).norm();
                impulses_r_af_xy[i] = impulses_r_af_vector.segment<2>(i*3).norm();
                impulses_r_af_z[i] = impulses_r_af_vector[i*3+2];
                if(impulses_r_af[i] > 0.01){
                    contacts_r_af[i] = 1;
                }
                else{
                    contacts_r_af[i] = 0;
                }
            }

            for(auto& contact_non_af: mano_r_->getContacts()) {
                if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                if (contact_non_af.getImpulse().e().norm() < 0.001) continue;
                int idx = contactMapping_r_[contact_non_af.getlocalBodyIndex()];
                Eigen::Vector3d impulse = contact_non_af.getContactFrame().e().transpose() * contact_non_af.getImpulse().e();
                if (!contact_non_af.isObjectA()) impulse = -impulse;
                impulses_r_non_af_vector[idx*3] += impulse[0];
                impulses_r_non_af_vector[idx*3+1] += impulse[1];
                impulses_r_non_af_vector[idx*3+2] += impulse[2];
            }

            for(int i = 0; i < num_contacts; i++){
                impulses_r_non_af[i] = impulses_r_non_af_vector.segment<3>(i*3).norm();
                if(impulses_r_non_af[i] > 0.01){
                    contacts_r_non_af[i] = 1;
                }
                else{
                    contacts_r_non_af[i] = 0;
                }
            }

            for(auto& contact_table: mano_r_->getContacts()) {
                if (contact_table.skip() || contact_table.getPairObjectIndex() != box->getIndexInWorld()) continue;
                if (contact_table.getImpulse().e().norm() < 0.001) continue;
                int idx = contactMapping_r_[contact_table.getlocalBodyIndex()];
                Eigen::Vector3d impulse = contact_table.getContactFrame().e().transpose() * contact_table.getImpulse().e();
                if (!contact_table.isObjectA()) impulse = -impulse;
                impulses_r_table_vector[idx*3] += impulse[0];
                impulses_r_table_vector[idx*3+1] += impulse[1];
                impulses_r_table_vector[idx*3+2] += impulse[2];
            }

            for(int i = 0; i < num_contacts; i++){
                impulses_r_table[i] = impulses_r_table_vector.segment<3>(i*3).norm();
                if(impulses_r_table[i] > 0.01){
                    contacts_r_table[i] = 1;
                }
                else{
                    contacts_r_table[i] = 0;
                }
            }   
            
            for(auto& contact_arm: mano_r_->getContacts()) {
                if ((contact_arm.skip() || contact_arm.getPairObjectIndex() != arctic->getIndexInWorld()) && 
                    (contact_arm.skip() || contact_arm.getPairObjectIndex() != box->getIndexInWorld())) continue;
                if (contact_arm.getImpulse().e().norm() < 0.001) continue;
                int idx = contactMapping_arm_[contact_arm.getlocalBodyIndex()];
                Eigen::Vector3d impulse = contact_arm.getContactFrame().e().transpose() * contact_arm.getImpulse().e();
                if (!contact_arm.isObjectA()) impulse = -impulse;
                impulses_arm_table_vector[idx*3] += impulse[0];
                impulses_arm_table_vector[idx*3+1] += impulse[1];
                impulses_arm_table_vector[idx*3+2] += impulse[2];
            }

            for(int i = 0; i < 6; i++){
                impulses_arm_table[i] = impulses_arm_table_vector.segment<3>(i*3).norm();
                if(impulses_arm_table[i] > 0.01){
                    contacts_arm_table[i] = 1;
                }
                else{
                    contacts_arm_table[i] = 0;
                }
            }

            for(auto& contact_arm: mano_r_->getContacts()) {
                contacts_arm_all[contactMapping_arm_[contact_arm.getlocalBodyIndex()]] = 1;
            }

            for(int i=0; i<num_contacts; i++){
                if (contacts_r_non_af[i] == 1){
                    contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                    impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                }
            }

            if (!has_non_aff){
                contacts_r_non_af.setZero();
                impulses_r_non_af.setZero();
            }

            right_hand_torque = (pTarget_clipped_r - gc_r_);

            raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
            raisim::Mat<3,3> obj_rot_w;
            Eigen::Vector3d target_center, target_center_wrist, target_center_dif_world;
            arctic->getPosition(affordance_id, obj_pos_w);
            arctic->getOrientation(affordance_id, obj_rot_w);
            raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
            mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
            target_center[0] = afford_center_w[0] + obj_pos_w[0];
            target_center[1] = afford_center_w[1] + obj_pos_w[1];
            target_center[2] = afford_center_w[2] + obj_pos_w[2];

            Eigen::Vector3d hand_center_w, hand_center_robot;
            hand_center_w = wrist_mat_r.e() * hand_center;
            hand_center_w[0] += wrist_pos_w[0];
            hand_center_w[1] += wrist_pos_w[1];
            hand_center_w[2] += wrist_pos_w[2];

            target_center_dif_world = target_center - hand_center_w;
            target_center_dif = wrist_mat_r.e().transpose() * target_center_dif_world;

            raisim::transpose(Obj_orientation_temp, Obj_orientation);

            raisim::Mat<3,3> obj_pose_wrist_mat;
            raisim::Vec<3> obj_pose_wrist;
            raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
            raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);

            raisim::Mat<3,3> frame_mat;
            raisim::Vec<3> joint_pos_w, joint_pos_o;
            raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o;

            for(int i = 0; i < num_bodyparts ; i++){
                mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                joint_pos_in_world[i * 3] = joint_pos_w[0];
                joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];
                joint_height_w[i] = joint_pos_w[2] - 0.771;

                raisim::Vec<3>  joint_pos_o_temp;
                joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);

                joint_pos_in_obj[i * 3] = joint_pos_o[0];
                joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
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

            obDouble_r_ << gc_r_,
                           right_hand_torque,
                           contacts_r_af,
                           impulses_r_af,
                           joint_height_w,
                           arm_height_w,
                           hand_center_w,
                           euler_diff,
                           wrist_euler_current.e();
                           
            obs_history.push_back(obDouble_r_);

            raisim::Vec<3> obj_pose;
            raisim::RotmatToEuler(Obj_orientation_temp, obj_pose);
            raisim::Vec<3> hand_pose_trans;
            raisim::RotmatToEuler(wrist_mat_r_trans, hand_pose_trans);

            raisim::Vec<3> wrist_pos_obj_temp, wrist_pos_obj;
            wrist_pos_obj_temp[0] = wrist_pos_w[0] - obj_pos_w[0];
            wrist_pos_obj_temp[1] = wrist_pos_w[1] - obj_pos_w[1];
            wrist_pos_obj_temp[2] = wrist_pos_w[2] - obj_pos_w[2];
            raisim::matvecmul(Obj_orientation, wrist_pos_obj_temp, wrist_pos_obj);

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
            // initialize: when first observe, only one observation is available, so fill (history_len-1) obs with the first obs
            if (vec_size == 1)
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

        void switch_obj_pos(Eigen::Ref<EigenVec> obj_pos_bias) {
            Eigen::Vector3d obj_pos_bias_temp;
            obj_pos_bias_temp = obj_pos_bias.cast<double>();
            obj_base_pos.setZero();
            arctic->getBasePosition(obj_base_pos);
            obj_base_pos[0] += obj_pos_bias_temp[0];
            obj_base_pos[1] += obj_pos_bias_temp[1];
            arctic->setBasePos(obj_base_pos);
            updateObservation();
        }

        /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
        bool isTerminalState(float& terminalReward) final {
            raisim::Vec<3> obj_current_pos;
            auto top_id = arctic->getBodyIdx("top");
            arctic->getPosition(top_id, obj_current_pos);

            for(int i = 0; i < num_bodyparts ; i++){
                if (joint_height_w[i] < -0.0){
                    terminalReward = -10;
                    std::cout<<"joint_height_w: "<<joint_height_w[i]<<std::endl;
                    return true;
                }
            }

            if(obDouble_r_.hasNaN() || global_state_.hasNaN())
            {
                std::cout<<"obj name: "<<obj_name<<std::endl;
                if (obDouble_r_.hasNaN()) std::cout<<"NaN detected obdouble"<< obDouble_r_.transpose()<<std::endl<<std::endl<<std::endl;
                if (global_state_.hasNaN()) std::cout<<"NaN detected global"<< global_state_.transpose()<<std::endl<<std::endl<<std::endl;
                return true;
            }

            return false;
        }

    private:
        int gcDim_, gvDim_, tobeEncode_dim, history_len, obDim_single;
        int gcDim_obj, gvDim_obj;
        bool visualizable_ = false;
        bool lift = false;
        int lift_num = 0;
        Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, vTarget_r_, gc_set_r_, gv_set_r_;
        Eigen::VectorXd obj_pos_init_;
        Eigen::VectorXd joint_pos_in_world;
        Eigen::VectorXd arm_joint_pos_in_world;
        std::string load_set;

        double affordance_contact_reward_r= 0.0;
        double not_affordance_contact_reward_r = 0.0;
        double table_contact_reward_r = 0.0;
        double affordance_impulse_reward_r= 0.0;
        double not_affordance_impulse_reward_r = 0.0;
        double push_reward_r = 0.0;
        double table_impulse_reward_r = 0.0;
        double wrist_vel_reward_r = 0.0;
        double wrist_qvel_reward_r = 0.0;
        double obj_vel_reward_r = 0.0;
        double obj_qvel_reward_r = 0.0;
        double arm_table_contact_reward = 0.0;
        double arm_table_impulse_reward = 0.0;
        double obj_displacement_reward = 0.0;
        double arm_joint_vel_reward = 0.0;
        double obj_weight = 0.0;

        std::string obj_name;

        int num_contacts = 0;
        int num_bodyparts = 0;

        raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
        raisim::Vec<3> init_root_r_, init_obj_;
        Eigen::VectorXd joint_limit_high, joint_limit_low;
        Eigen::VectorXd impulse_high, impulse_low;
        Eigen::VectorXd actionMean_r_, actionStd_r_, arm_gc_lift;
        Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_, ob_delay_r, ob_concat_r;
        Eigen::VectorXd finger_weights_contact;
        Eigen::VectorXd contacts_r_af, impulses_r_af;
        Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
        Eigen::VectorXd contacts_r_table, impulses_r_table;
        Eigen::VectorXd contacts_arm_table, impulses_arm_table, contacts_arm_all;
        Eigen::VectorXd contact_body_idx_r_, contact_arm_idx;
        Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj, joint_height_w, arm_height_w;
        raisim::Vec<3> Position;
        raisim::Vec<3> wrist_vel, wrist_qvel;
        raisim::Vec<3> obj_base_pos;
        Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
        Eigen::VectorXd right_hand_torque;
        Eigen::VectorXd pTarget_clipped_r, pTarget_prev_r;
        Eigen::Vector3d hand_center, afford_center;
        std::deque<Eigen::VectorXd> obs_history;

        Eigen::VectorXd init_arcticCoord;

        raisim::Box *box;
        raisim::ArticulatedSystem *arctic;
        std::unique_ptr<Hardware> mano_r_;
        raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
        raisim::Vec<3> wrist_euler_in_obj_init, wrist_euler_init;
        raisim::Vec<4> obj_quat;
        raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
        Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
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
        raisim::Visuals *sphere[17];
        raisim::Visuals *joints_sphere[17];
        raisim::Visuals *aff_center_visual[7];
        raisim::Visuals *wrist_target[2];

        raisim::Vec<3> base_pos;
        raisim::Mat<3,3> base_mat;
        raisim::Mat<3,3> wrist_mat_r_init;
        raisim::Vec<3> wrist_euler_previous;

        std::map<int,int> contactMapping_r_;
        std::map<int,int> contactMapping_arm_;
        std::string resourceDir_;
        std::vector<raisim::Vec<2>> joint_limits_;

        Eigen::VectorXd impulses_r_af_vector, impulses_r_non_af_vector, impulses_r_table_vector, impulses_arm_table_vector, impulses_r_af_xy, impulses_r_af_z;
    };
}