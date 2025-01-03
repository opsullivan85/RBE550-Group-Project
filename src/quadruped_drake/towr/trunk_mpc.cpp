#include <cmath>
#include <iostream>

// #include <towr/terrain/examples/height_map_examples.h>
#include <towr/terrain/height_map.h>
#include <towr/terrain/Grid.h>
#include <towr/nlp_formulation.h>
#include <ifopt/ipopt_solver.h>

#include <towr/initialization/gait_generator.h>
#include <towr/models/endeffector_mappings.h>
#include <Eigen/Dense>

#include <lcm/lcm-cpp.hpp>
#include "../lcm_types/trunklcm/trunk_state_t.hpp"

using namespace towr;

// Publish the trajectory contained in the given solution over lcm
void publish_trunk_state(SplineHolder solution, double t, bool finished = false)
{

    lcm::LCM lcm;
    trunklcm::trunk_state_t state;

    state.timestamp = t;
    state.finished = finished;

    // Base linear position/vel/accel
    Eigen::Map<Eigen::VectorXd>(&state.base_p[0], 3) = solution.base_linear_->GetPoint(t).p();
    Eigen::Map<Eigen::VectorXd>(&state.base_pd[0], 3) = solution.base_linear_->GetPoint(t).v();
    Eigen::Map<Eigen::VectorXd>(&state.base_pdd[0], 3) = solution.base_linear_->GetPoint(t).a();

    // Base angular position/vel/accel
    Eigen::Map<Eigen::VectorXd>(&state.base_rpy[0], 3) = solution.base_angular_->GetPoint(t).p();
    Eigen::Map<Eigen::VectorXd>(&state.base_rpyd[0], 3) = solution.base_angular_->GetPoint(t).v();
    Eigen::Map<Eigen::VectorXd>(&state.base_rpydd[0], 3) = solution.base_angular_->GetPoint(t).a();

    // Foot positions
    Eigen::Map<Eigen::VectorXd>(&state.lf_p[0], 3) = solution.ee_motion_.at(LF)->GetPoint(t).p();
    Eigen::Map<Eigen::VectorXd>(&state.rf_p[0], 3) = solution.ee_motion_.at(RF)->GetPoint(t).p();
    Eigen::Map<Eigen::VectorXd>(&state.lh_p[0], 3) = solution.ee_motion_.at(LH)->GetPoint(t).p();
    Eigen::Map<Eigen::VectorXd>(&state.rh_p[0], 3) = solution.ee_motion_.at(RH)->GetPoint(t).p();

    // Foot velocities
    Eigen::Map<Eigen::VectorXd>(&state.lf_pd[0], 3) = solution.ee_motion_.at(LF)->GetPoint(t).v();
    Eigen::Map<Eigen::VectorXd>(&state.rf_pd[0], 3) = solution.ee_motion_.at(RF)->GetPoint(t).v();
    Eigen::Map<Eigen::VectorXd>(&state.lh_pd[0], 3) = solution.ee_motion_.at(LH)->GetPoint(t).v();
    Eigen::Map<Eigen::VectorXd>(&state.rh_pd[0], 3) = solution.ee_motion_.at(RH)->GetPoint(t).v();

    // Foot accelerations
    Eigen::Map<Eigen::VectorXd>(&state.lf_pdd[0], 3) = solution.ee_motion_.at(LF)->GetPoint(t).a();
    Eigen::Map<Eigen::VectorXd>(&state.rf_pdd[0], 3) = solution.ee_motion_.at(RF)->GetPoint(t).a();
    Eigen::Map<Eigen::VectorXd>(&state.lh_pdd[0], 3) = solution.ee_motion_.at(LH)->GetPoint(t).a();
    Eigen::Map<Eigen::VectorXd>(&state.rh_pdd[0], 3) = solution.ee_motion_.at(RH)->GetPoint(t).a();

    // Foot contact states
    state.lf_contact = solution.phase_durations_.at(LF)->IsContactPhase(t);
    state.rf_contact = solution.phase_durations_.at(RF)->IsContactPhase(t);
    state.lh_contact = solution.phase_durations_.at(LH)->IsContactPhase(t);
    state.rh_contact = solution.phase_durations_.at(RH)->IsContactPhase(t);

    // Foot contact forces
    Eigen::Map<Eigen::VectorXd>(&state.lf_f[0], 3) = solution.ee_force_.at(LF)->GetPoint(t).p();
    Eigen::Map<Eigen::VectorXd>(&state.rf_f[0], 3) = solution.ee_force_.at(RF)->GetPoint(t).p();
    Eigen::Map<Eigen::VectorXd>(&state.lh_f[0], 3) = solution.ee_force_.at(LH)->GetPoint(t).p();
    Eigen::Map<Eigen::VectorXd>(&state.rh_f[0], 3) = solution.ee_force_.at(RH)->GetPoint(t).p();

    lcm.publish("trunk_state", &state);
}

// Generate a trunk-model trajectory for a quadruped using TOWR, and send the results
// over LCM, where they can be read by Drake.
int main(int argc, char *argv[])
{

    // Command line argument parsing
    // fl_x, br_z, etc. are front left foot x position, back right foot z position, etc.
    char usage_message[] = "Usage: trunk_mpc gait_type={walk,trot,pace,bound,gallop} optimize_gait={0,1} x_init, y_init, yaw_init, x_final, y_final, yaw_final, grid.csv, fl_x, fl_y, fl_z, fr_x, fr_y, fr_z, bl_x, bl_y, bl_z, br_x, br_y, br_z, trunk_z_init, trunk_z_final, roll_init, pitch_init, roll_final, pitch_final, duration";
    if (argc != 29)
    {
        std::cout << usage_message << std::endl;
        return 1;
    }

    int gait_type;
    if (!strcmp(argv[1], "walk"))
    {
        gait_type = 0;
    }
    else if (!strcmp(argv[1], "trot"))
    {
        gait_type = 1;
    }
    else if (!strcmp(argv[1], "pace"))
    {
        gait_type = 2;
    }
    else if (!strcmp(argv[1], "bound"))
    {
        gait_type = 3;
    }
    else if (!strcmp(argv[1], "gallop"))
    {
        gait_type = 4;
    }
    else
    {
        std::cout << "Invalid gait_type " << argv[1] << std::endl;
        std::cout << usage_message << std::endl;
        return 1;
    }

    bool optimize_gait = !strcmp(argv[2], "1");

    float x_init = std::stof(argv[3]);
    float y_init = std::stof(argv[4]);
    float yaw_init = std::stof(argv[5]);
    float x_final = std::stof(argv[6]);
    float y_final = std::stof(argv[7]);
    float yaw_final = std::stof(argv[8]);
    const std::string grid_csv = std::string(argv[9]);
    float fl_x = std::stof(argv[10]);
    float fl_y = std::stof(argv[11]);
    float fl_z = std::stof(argv[12]);
    float fr_x = std::stof(argv[13]);
    float fr_y = std::stof(argv[14]);
    float fr_z = std::stof(argv[15]);
    float bl_x = std::stof(argv[16]);
    float bl_y = std::stof(argv[17]);
    float bl_z = std::stof(argv[18]);
    float br_x = std::stof(argv[19]);
    float br_y = std::stof(argv[20]);
    float br_z = std::stof(argv[21]);
    float trunk_z_init = std::stof(argv[22]);
    float trunk_z_final = std::stof(argv[23]);
    float roll_init = std::stof(argv[24]);
    float pitch_init = std::stof(argv[25]);
    float roll_final = std::stof(argv[26]);
    float pitch_final = std::stof(argv[27]);
    float total_duration = std::stof(argv[28]);

    // Set up the NLP
    NlpFormulation formulation;

    // terrain
    formulation.terrain_ = std::make_shared<Grid>(grid_csv);

    // Kinematic limits and dynamic parameters
    formulation.model_ = RobotModel(RobotModel::MiniCheetah);

    // initial position
    auto nominal_stance_B = formulation.model_.kinematic_model_->GetNominalStanceInBase();
    nominal_stance_B.at(LF) << fl_x, fl_y, fl_z;
    nominal_stance_B.at(RF) << fr_x, fr_y, fr_z;
    nominal_stance_B.at(LH) << bl_x, bl_y, bl_z;
    nominal_stance_B.at(RH) << br_x, br_y, br_z;
    // TODO: I beleive this is preventing the correct estimation of the body height
    double z_ground = 0.0;
    formulation.initial_ee_W_ = nominal_stance_B;
    // formulation.initial_base_.lin.at(kPos).z() = -nominal_stance_B.front().z() + z_ground;
    formulation.initial_base_.lin.at(towr::kPos) << x_init, y_init, trunk_z_init;
    formulation.initial_base_.ang.at(towr::kPos) << roll_init, pitch_init, yaw_init;

    // desired goal state
    // formulation.final_base_.lin.at(towr::kPos) << x_final, y_final, -nominal_stance_B.front().z() + z_ground;
    formulation.final_base_.lin.at(towr::kPos) << x_final, y_final, trunk_z_final;
    formulation.final_base_.ang.at(towr::kPos) << roll_final, pitch_final, yaw_final;

    // Parameters defining contact sequence and default durations. We use
    // a GaitGenerator with some predifined gaits
    auto gait_gen_ = GaitGenerator::MakeGaitGenerator(4);
    auto id_gait = static_cast<GaitGenerator::Combos>(gait_type); // 0=walk, 1=flying trot, 2=pace, 3=bound, 4=gallop
    gait_gen_->SetCombo(id_gait);
    for (int ee = 0; ee < 4; ++ee)
    {
        formulation.params_.ee_phase_durations_.push_back(gait_gen_->GetPhaseDurations(total_duration, ee));
        formulation.params_.ee_in_contact_at_start_.push_back(gait_gen_->IsInContactAtStart(ee));
    }

    // Indicate whether to optimize over gaits as well
    if (optimize_gait)
    {
        formulation.params_.OptimizePhaseDurations();
    }

    // Add weighted cost on rotational velocity of base
    // formulation.params_.costs_.push_back({Parameters::CostName(1),1.0});

    // Initialize the nonlinear-programming problem with the variables,
    // constraints and costs.
    ifopt::Problem nlp;
    SplineHolder solution;
    for (auto c : formulation.GetVariableSets(solution))
        nlp.AddVariableSet(c);
    for (auto c : formulation.GetConstraints(solution))
        nlp.AddConstraintSet(c);
    for (auto c : formulation.GetCosts())
        nlp.AddCostSet(c);

    // Choose ifopt solver (IPOPT or SNOPT), set some parameters and solve.
    // solver->SetOption("derivative_test", "first-order");
    auto solver = std::make_shared<ifopt::IpoptSolver>();
    solver->SetOption("jacobian_approximation", "exact"); // "finite difference-values"
    solver->SetOption("hessian_approximation", "limited-memory");
    solver->SetOption("acceptable_iter", 15);
    solver->SetOption("acceptable_tol", 1e-3);
    solver->SetOption("max_iter", 500);
    solver->SetOption("max_cpu_time", 8.0);
    solver->SetOption("tol", 1e-4);
    solver->Solve(nlp);

    // Send solution over LCM
    lcm::LCM lcm;
    trunklcm::trunk_state_t state;

    double dt = 1e-3;
    for (double t = 0; t < total_duration; t = t + dt)
    {
        publish_trunk_state(solution, t);
    }

    // send one final message including the finished flag
    publish_trunk_state(solution, total_duration, true);
}
