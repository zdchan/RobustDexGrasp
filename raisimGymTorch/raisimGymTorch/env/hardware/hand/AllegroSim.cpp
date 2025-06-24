#include "../hardwareHand.hpp"

// raisim library
#include "raisim/World.hpp"
#include "raisim/math.hpp"
#include <time.h> 

class AllegroSim : public HardwareHand {
public:
    void init(const std::string &rsc_pth, const Yaml::Node &cfg) final override {
        wrist_pose_.setZero(6);
        wrist_velocity_.setZero(6);
        hand_joint_position_.setZero(num_joint_);
        hand_joint_velocity_.setZero(num_joint_);
        
        flying_hand_mode_ = cfg["flying_hand_mode"].As<bool>();
        if (!cfg["randomize_gc_hand"].IsNone()) {
            randomize_gc_ = cfg["randomize_gc_hand"].As<double>();
        }
        std::string pd_file = cfg["hand_pd_file"].As<std::string>();

        std::ifstream pd_txt;
        pd_txt.open(rsc_pth+"/../raisimGymTorch/raisimGymTorch/env/hardware/hand/"+pd_file);
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
        int all_joint_num = joint_limits.size();
        for(int i = 0; i < num_joint_; i++){
            joint_limit_low_[i] = joint_limits[all_joint_num - num_joint_ + i][0];
            joint_limit_high_[i] = joint_limits[all_joint_num - num_joint_ + i][1];
        }
    }

    void updateHandState(const Eigen::VectorXd &eef_pos) final override {
        wrist_pose_ = eef_pos;
        Eigen::VectorXd gc(platform_->getGeneralizedCoordinateDim()), gv(platform_->getDOF());
        platform_->getState(gc, gv);
        hand_joint_position_ = gc.tail(num_joint_);
        hand_joint_velocity_ = gv.tail(num_joint_);
        if (randomize_gc_ > 1e-9) {
            hand_joint_position_ += Eigen::VectorXd::Random(num_joint_) * randomize_gc_;
            hand_joint_position_ = hand_joint_position_.cwiseMax(joint_limit_low_).cwiseMin(joint_limit_high_);
        }

    }
    void setPdTarget(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget, bool async = true) final override {
        platform_->setPdTarget(posTarget, velTarget);
    }

    void getPdgains(Eigen::VectorXd &pgain, Eigen::VectorXd &dgain, int tail_shift) const final override {
        for (int i = 0; i < num_joint_; i++) {
            pgain.tail(tail_shift)[i] = Pgain[i];
            dgain.tail(tail_shift)[i] = Dgain[i];
        }
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
                if (flying_hand_mode_) {
                    get_vec.push_back(body_parts_flying_[i]);
                } else {
                    get_vec.push_back(body_parts_[i]);
                }
            }
            return num_bodies_;
        }
    }

    std::string changeJointToLinkName(std::string frameName) const final override {
        if (!frameName.compare("Flange2hand_fixed_joint")) {
            return std::string("Flange_base_link");
        } else if (!frameName.compare("joint_3.0_tip")) {
            return std::string("link_3.0_tip");
        } else if (!frameName.compare("joint_7.0_tip")) {
            return std::string("link_7.0_tip");
        } else if (!frameName.compare("joint_11.0_tip")) {
            return std::string("link_11.0_tip");
        } else if (!frameName.compare("joint_15.0_tip")) {
            return std::string("link_15.0_tip");
        } else {
            return frameName;
        }
    }

private:
    raisim::ArticulatedSystem *platform_;

    bool flying_hand_mode_ = false;
    double randomize_gc_ = 0.0;
    Eigen::VectorXd joint_limit_high_;
    Eigen::VectorXd joint_limit_low_;

    const static int num_contacts_ = 13;
    const static int num_bodies_ = 17;
    const static int num_finger_ = 4;
    const static int num_joint_ = 16;

    double Pgain[num_joint_] = {60.0};
    double Dgain[num_joint_] = {0.2};

    const std::string body_parts_flying_[num_bodies_] =  {"z_rotation_joint",
    "joint_1.0", "joint_2.0", "joint_3.0", "joint_3.0_tip",
    "joint_5.0", "joint_6.0", "joint_7.0", "joint_7.0_tip",
    "joint_9.0", "joint_10.0", "joint_11.0", "joint_11.0_tip",
    "joint_13.0", "joint_14.0", "joint_15.0", "joint_15.0_tip"};

    const std::string body_parts_[num_bodies_] =  {"Flange2hand_fixed_joint",
    "joint_1.0", "joint_2.0", "joint_3.0", "joint_3.0_tip",
    "joint_5.0", "joint_6.0", "joint_7.0", "joint_7.0_tip",
    "joint_9.0", "joint_10.0", "joint_11.0", "joint_11.0_tip",
    "joint_13.0", "joint_14.0", "joint_15.0", "joint_15.0_tip"};

    // for raisim contact check
    const std::string contact_bodies_[num_contacts_] =  {"wrist_3_link",
    "link_1.0", "link_2.0", "link_3.0",
    "link_5.0", "link_6.0", "link_7.0",
    "link_9.0", "link_10.0", "link_11.0",
    "link_13.0", "link_14.0", "link_15.0"};
};

extern "C" std::unique_ptr<HardwareHand> createAllegroSim() {
    return std::make_unique<AllegroSim>();
}