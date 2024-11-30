#ifndef GRID_H_
#define GRID_H_

#include <iostream>
#include "height_map.h"
#include <Eigen/Dense>
#include <rapidcsv.h>

class Grid final : public towr::HeightMap
{
public:
    Grid(const std::string& file_path)
    {
        // open csv file and convert it to an eigen matrix
        rapidcsv::Document doc(file_path, rapidcsv::LabelParams(-1,-1));
        grid_ = Eigen::MatrixXd::Zero(doc.GetRowCount(), doc.GetColumnCount());
        for (size_t i{}; i<doc.GetRowCount(); ++i){
            //const auto row = doc.GetRow<double>(i);
            //std::cout << "row " << i << std::endl;
            for (size_t j{}; j<doc.GetColumnCount(); ++j){
                //grid_(i, j) = row.at(j);
                // std::cout << "col " << j << std::endl;
                // std::cout << "cell value: " << doc.GetCell<std::string>(j, i);
                grid_(i, j) = doc.GetCell<double>(j, i);
            }
        }
    };

    double GetHeight(double x, double y) const override
    {
        const size_t x_cell = static_cast<size_t>(x/res_m_p_cell_);
        const size_t y_cell = static_cast<size_t>(y/res_m_p_cell_);
        if ((x_cell >= grid_.cols()) || (y_cell >= grid_.rows())){
            return 0.0;
        }
        return grid_(y_cell, x_cell);
    };

    double GetHeightDerivWrtX(double x, double y) const override
    {
        const size_t x_cell = static_cast<size_t>(x/res_m_p_cell_);
        const size_t y_cell = static_cast<size_t>(y/res_m_p_cell_);
        if ((x_cell+1 >= grid_.cols()) || (y_cell >= grid_.rows())){
            return 0.0;
        }

        const double diff = grid_(y_cell, x_cell+1) - grid_(y_cell, x_cell);

        const double x_start_cell = (x_cell + 1) * res_m_p_cell_;
        if ((x <= (x_start_cell + eps_/2)) && (x >= x_start_cell - eps_/2)){
            return diff / eps_;
        }
        return 0.0;
    }

    double GetHeightDerivWrtY(double x, double y) const override
    {
        const size_t x_cell = static_cast<size_t>(x/res_m_p_cell_);
        const size_t y_cell = static_cast<size_t>(y/res_m_p_cell_);
        if ((x_cell >= grid_.cols()) || (y_cell+1 >= grid_.rows())){
            return 0.0;
        }
        
        const double diff = grid_(y_cell+1, x_cell) - grid_(y_cell, x_cell);

        const double y_start_cell = (y_cell + 1) * res_m_p_cell_;
        if ((y <= (y_start_cell + eps_/2)) && (y >= y_start_cell - eps_/2)){
            return diff / eps_;
        }
        return 0.0;
    }

  private:
    Eigen::MatrixXd grid_;
    const double res_m_p_cell_ = 0.17;
    const double eps_ = 0.03;
  };

#endif
