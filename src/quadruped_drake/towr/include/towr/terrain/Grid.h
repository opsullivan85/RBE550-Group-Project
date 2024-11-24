#ifndef GRID_H_
#define GRID_H_

#include "height_map.h"
#include <Eigen/Dense>

class Grid final : public HeightMap
{
public:
    Grid(const std::string& file_path) : plant(0.0)
    {
        // open csv file and convert it to an eigen matrix
      

        drake::geometry::SceneGraph<double> scene_graph;
        drake::multibody::Parser parser(&(this->plant), &scene_graph);
        auto model_instances = parser.AddModelsFromUrl(std::string("file://").append(file_path));
        if (model_instances.size() != 1)
            {
                throw std::runtime_error("Expected exactly one model instance.");
            }
        this->model_instance = model_instances[0];
        this->plant.Finalize();
    };

    double GetHeight(double x, double y) const override
    {
        // return std::max(std::floor(x)/10, 0);
        double height = 0.0;

        // Loop through each body in the plant
        for (const auto &body_index : this->plant.GetBodyIndices(this->model_instance))
            {
                const auto &body = this->plant.get_body(body_index);

                // Assuming rectangular prisms are fixed, get their pose in the world frame
                const auto &X_WB = this->plant.EvalBodyPoseInWorld(*(this->plant).CreateDefaultContext(), body);

                // Extract the position of the body
                Eigen::Vector3d position = X_WB.translation();

                // Check if the coordinate falls within this body's footprint
                const double prism_width = 0.5;
                const double prism_depth = 0.5;

                if (x >= position.x() - prism_width / 2 &&
                    x <= position.x() + prism_width / 2 &&
                    y >= position.y() - prism_depth / 2 &&
                    y <= position.y() + prism_depth / 2)
                    {
                        height = 2 * position.z(); // Assuming the prism’s top corresponds to the height
                        break;
                    }
            }

        return height;
    };


  private:
    // std::vector<std::vector<double>> height_map;
    drake::multibody::MultibodyPlant<double> plant;
    drake::multibody::ModelInstanceIndex model_instance;
  };

#endif
