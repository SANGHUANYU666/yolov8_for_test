// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// 1. install
//      pip3 install -U ultralytics pnnx ncnn
// 2. export yolov8 torchscript
//      yolo export model=yolov8n.pt format=torchscript
// 3. convert torchscript with static shape
//      pnnx yolov8n.torchscript
// 4. modify yolov8n_pnnx.py for dynamic shape inference
//      A. modify reshape to support dynamic image sizes
//      B. permute tensor before concat and adjust concat axis
//      C. drop post-process part
//      before:
//          v_165 = v_142.view(1, 144, 6400)
//          v_166 = v_153.view(1, 144, 1600)
//          v_167 = v_164.view(1, 144, 400)
//          v_168 = torch.cat((v_165, v_166, v_167), dim=2)
//          ...
//      after:
//          v_165 = v_142.view(1, 144, -1).transpose(1, 2)
//          v_166 = v_153.view(1, 144, -1).transpose(1, 2)
//          v_167 = v_164.view(1, 144, -1).transpose(1, 2)
//          v_168 = torch.cat((v_165, v_166, v_167), dim=1)
//          return v_168
// 5. re-export yolov8 torchscript
//      python3 -c 'import yolov8n_pnnx; yolov8n_pnnx.export_torchscript()'
// 6. convert new torchscript with dynamic shape
//      pnnx yolov8n_pnnx.py.pt inputshape=[1,3,640,640] inputshape2=[1,3,320,320]
// 7. now you get ncnn model files
//      mv yolov8n_pnnx.py.ncnn.param yolov8n.ncnn.param
//      mv yolov8n_pnnx.py.ncnn.bin yolov8n.ncnn.bin

// the out blob would be a 2-dim tensor with w=144 h=8400
//
//        | bbox-reg 16 x 4       | per-class scores(80) |
//        +-----+-----+-----+-----+----------------------+
//        | dx0 | dy0 | dx1 | dy1 |0.1 0.0 0.0 0.5 ......|
//   all /|     |     |     |     |           .          |
//  boxes |  .. |  .. |  .. |  .. |0.0 0.9 0.0 0.0 ......|
//  (8400)|     |     |     |     |           .          |
//       \|     |     |     |     |           .          |
//        +-----+-----+-----+-----+----------------------+
//

#include "yolov8.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "BYTETracker.h" //bytetrack

class PortGridReconstructor
{
public:
    PortGridReconstructor()
    {
        reset();
    }

    void update(const std::vector<Object>& objects)
    {
        frame_index++;

        std::vector<PortDetection> detections;
        detections.reserve(objects.size());
        for (size_t i = 0; i < objects.size(); i++)
        {
            const Object& obj = objects[i];
            if (obj.label < 0 || obj.label > 1 || obj.prob < 0.20f)
                continue;

            PortDetection det;
            det.pt = cv::Point2f(obj.rect.x + obj.rect.width * 0.5f, obj.rect.y + obj.rect.height * 0.5f);
            det.rect = obj.rect;
            det.state = obj.label;
            det.conf = obj.prob;
            detections.push_back(det);
        }

        if (detections.empty())
            return;

        if (!has_global_grid)
            try_init_global_grid(detections);

        LocalGrid local_grid;
        if (!fit_local_grid(detections, local_grid))
            return;

        int row0 = 0;
        int col0 = 0;
        if (!match_local_grid(local_grid, row0, col0))
            return;

        has_current_window = true;
        current_row0 = row0;
        current_col0 = col0;
        current_rows = local_grid.rows;
        current_cols = local_grid.cols;

        for (size_t i = 0; i < local_grid.items.size(); i++)
        {
            const LocalItem& item = local_grid.items[i];
            int row = row0 + item.local_row;
            int col = col0 + item.local_col;
            if (row < 0 || row >= kRows || col < 0 || col >= kCols)
                continue;

            update_cell(row, col, item.det.state, item.det.conf);
        }
    }

    void draw(cv::Mat& rgb) const
    {
        draw_status_panel(rgb);

        if (!has_current_window)
            return;

        char text[96];
        snprintf(text, sizeof(text), "ports r%d-%d c%d-%d",
                 current_row0 + 1, current_row0 + current_rows,
                 current_col0 + 1, current_col0 + current_cols);
        cv::putText(rgb, text, cv::Point(8, 132), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    std::string build_json() const
    {
        std::string json;
        json.reserve(700);
        json += "\"portGridReady\":";
        json += has_global_grid ? "true" : "false";
        json += ",\"portWindow\":{\"row0\":";
        json += std::to_string(has_current_window ? current_row0 : -1);
        json += ",\"col0\":";
        json += std::to_string(has_current_window ? current_col0 : -1);
        json += ",\"rows\":";
        json += std::to_string(has_current_window ? current_rows : 0);
        json += ",\"cols\":";
        json += std::to_string(has_current_window ? current_cols : 0);
        json += "},\"portMatrix\":[";

        for (int r = 0; r < kRows; r++)
        {
            if (r > 0)
                json += ",";
            json += "[";
            for (int c = 0; c < kCols; c++)
            {
                if (c > 0)
                    json += ",";
                json += std::to_string(cells[r][c].state);
            }
            json += "]";
        }

        json += "]";
        return json;
    }

private:
    static const int kRows = 12;
    static const int kCols = 12;

    struct PortCell
    {
        int state;
        float conf;
        int frame;
    };

    struct PortDetection
    {
        cv::Point2f pt;
        cv::Rect_<float> rect;
        int state;
        float conf;
    };

    struct LocalItem
    {
        PortDetection det;
        int local_row;
        int local_col;
    };

    struct LocalGrid
    {
        std::vector<LocalItem> items;
        int rows;
        int cols;
    };

    PortCell cells[kRows][kCols];
    cv::Point2f global_points[kRows][kCols];
    bool has_global_grid;
    bool has_current_window;
    int current_row0;
    int current_col0;
    int current_rows;
    int current_cols;
    int frame_index;

    void reset()
    {
        has_global_grid = false;
        has_current_window = false;
        current_row0 = 0;
        current_col0 = 0;
        current_rows = 0;
        current_cols = 0;
        frame_index = 0;
        for (int r = 0; r < kRows; r++)
        {
            for (int c = 0; c < kCols; c++)
            {
                cells[r][c].state = -1;
                cells[r][c].conf = 0.f;
                cells[r][c].frame = 0;
                global_points[r][c] = cv::Point2f(0.f, 0.f);
            }
        }
    }

    static float median(std::vector<float> values, float fallback)
    {
        if (values.empty())
            return fallback;
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    static std::vector<float> cluster_axis(std::vector<float> values, float threshold)
    {
        std::vector<float> centers;
        if (values.empty())
            return centers;

        std::sort(values.begin(), values.end());

        float sum = values[0];
        int count = 1;
        float center = values[0];
        for (size_t i = 1; i < values.size(); i++)
        {
            if (std::fabs(values[i] - center) <= threshold)
            {
                sum += values[i];
                count++;
                center = sum / count;
            }
            else
            {
                centers.push_back(center);
                sum = values[i];
                count = 1;
                center = values[i];
            }
        }

        centers.push_back(center);
        return centers;
    }

    static int nearest_cluster(const std::vector<float>& centers, float value)
    {
        int best = 0;
        float best_dist = std::numeric_limits<float>::max();
        for (size_t i = 0; i < centers.size(); i++)
        {
            float dist = std::fabs(value - centers[i]);
            if (dist < best_dist)
            {
                best_dist = dist;
                best = (int)i;
            }
        }
        return best;
    }

    bool fit_local_grid(const std::vector<PortDetection>& detections, LocalGrid& grid) const
    {
        if (detections.size() < 4)
            return false;

        std::vector<float> xs;
        std::vector<float> ys;
        std::vector<float> widths;
        std::vector<float> heights;
        xs.reserve(detections.size());
        ys.reserve(detections.size());
        widths.reserve(detections.size());
        heights.reserve(detections.size());

        for (size_t i = 0; i < detections.size(); i++)
        {
            xs.push_back(detections[i].pt.x);
            ys.push_back(detections[i].pt.y);
            widths.push_back(detections[i].rect.width);
            heights.push_back(detections[i].rect.height);
        }

        float x_threshold = std::max(8.f, median(widths, 16.f) * 0.85f);
        float y_threshold = std::max(8.f, median(heights, 16.f) * 0.85f);
        std::vector<float> col_centers = cluster_axis(xs, x_threshold);
        std::vector<float> row_centers = cluster_axis(ys, y_threshold);

        if (row_centers.empty() || col_centers.empty() || row_centers.size() > kRows || col_centers.size() > kCols)
            return false;

        grid.items.clear();
        grid.rows = (int)row_centers.size();
        grid.cols = (int)col_centers.size();
        for (size_t i = 0; i < detections.size(); i++)
        {
            LocalItem item;
            item.det = detections[i];
            item.local_row = nearest_cluster(row_centers, detections[i].pt.y);
            item.local_col = nearest_cluster(col_centers, detections[i].pt.x);
            grid.items.push_back(item);
        }

        return true;
    }

    void try_init_global_grid(const std::vector<PortDetection>& detections)
    {
        if (detections.size() < 36)
            return;

        float min_x = detections[0].pt.x;
        float max_x = detections[0].pt.x;
        float min_y = detections[0].pt.y;
        float max_y = detections[0].pt.y;
        for (size_t i = 1; i < detections.size(); i++)
        {
            min_x = std::min(min_x, detections[i].pt.x);
            max_x = std::max(max_x, detections[i].pt.x);
            min_y = std::min(min_y, detections[i].pt.y);
            max_y = std::max(max_y, detections[i].pt.y);
        }

        if (max_x - min_x < 80.f || max_y - min_y < 80.f)
            return;

        for (int r = 0; r < kRows; r++)
        {
            float y = min_y + (max_y - min_y) * r / (kRows - 1);
            for (int c = 0; c < kCols; c++)
            {
                float x = min_x + (max_x - min_x) * c / (kCols - 1);
                global_points[r][c] = cv::Point2f(x, y);
            }
        }

        has_global_grid = true;
    }

    bool match_local_grid(const LocalGrid& local_grid, int& row0, int& col0)
    {
        if (!has_global_grid)
            return false;

        int max_row0 = kRows - local_grid.rows;
        int max_col0 = kCols - local_grid.cols;
        if (max_row0 < 0 || max_col0 < 0)
            return false;

        int search_row_min = 0;
        int search_row_max = max_row0;
        int search_col_min = 0;
        int search_col_max = max_col0;

        if (has_current_window)
        {
            search_row_min = std::max(0, current_row0 - 2);
            search_row_max = std::min(max_row0, current_row0 + 2);
            search_col_min = std::max(0, current_col0 - 2);
            search_col_max = std::min(max_col0, current_col0 + 2);
        }

        float best_score = std::numeric_limits<float>::max();
        int best_row = -1;
        int best_col = -1;
        for (int sr = search_row_min; sr <= search_row_max; sr++)
        {
            for (int sc = search_col_min; sc <= search_col_max; sc++)
            {
                float score = match_score(local_grid, sr, sc);
                if (score < best_score)
                {
                    best_score = score;
                    best_row = sr;
                    best_col = sc;
                }
            }
        }

        if (best_row < 0)
            return false;

        row0 = best_row;
        col0 = best_col;
        return true;
    }

    float match_score(const LocalGrid& local_grid, int row0, int col0) const
    {
        int known_count = 0;
        int state_mismatch = 0;
        float score = 0.f;

        for (size_t i = 0; i < local_grid.items.size(); i++)
        {
            const LocalItem& item = local_grid.items[i];
            int row = row0 + item.local_row;
            int col = col0 + item.local_col;

            if (cells[row][col].state >= 0)
            {
                known_count++;
                if (cells[row][col].state != item.det.state)
                    state_mismatch++;
            }
        }

        score += state_mismatch * 8.f;
        score -= known_count * 0.4f;

        if (has_current_window)
        {
            score += std::abs(row0 - current_row0) * 1.2f;
            score += std::abs(col0 - current_col0) * 1.2f;
        }

        return score;
    }

    void update_cell(int row, int col, int state, float conf)
    {
        PortCell& cell = cells[row][col];
        int age = frame_index - cell.frame;
        bool stale = age > 20;
        bool stronger = conf >= cell.conf + 0.05f;
        bool same_state = cell.state == state;

        if (cell.state < 0 || same_state || stale || stronger)
        {
            cell.state = state;
            cell.conf = same_state ? std::max(cell.conf * 0.90f, conf) : conf;
            cell.frame = frame_index;
        }
    }

    void draw_status_panel(cv::Mat& rgb) const
    {
        const int cell = 12;
        const int gap = 2;
        const int x0 = 8;
        const int y0 = std::max(150, rgb.rows - (cell + gap) * kRows - 18);

        cv::rectangle(rgb, cv::Rect(x0 - 4, y0 - 18, (cell + gap) * kCols + 8, (cell + gap) * kRows + 24),
                      cv::Scalar(0, 0, 0), -1);
        cv::putText(rgb, has_global_grid ? "12x12 ports" : "scan full cabinet",
                    cv::Point(x0, y0 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        for (int r = 0; r < kRows; r++)
        {
            for (int c = 0; c < kCols; c++)
            {
                cv::Scalar color(80, 80, 80);
                if (cells[r][c].state == 0)
                    color = cv::Scalar(60, 210, 80);
                else if (cells[r][c].state == 1)
                    color = cv::Scalar(50, 80, 240);

                int x = x0 + c * (cell + gap);
                int y = y0 + r * (cell + gap);
                cv::rectangle(rgb, cv::Rect(x, y, cell, cell), color, -1);

                if (has_current_window &&
                    r >= current_row0 && r < current_row0 + current_rows &&
                    c >= current_col0 && c < current_col0 + current_cols)
                {
                    cv::rectangle(rgb, cv::Rect(x, y, cell, cell), cv::Scalar(255, 255, 255), 1);
                }
            }
        }
    }
};

static PortGridReconstructor g_port_reconstructor;

std::string get_port_matrix_json()
{
    return g_port_reconstructor.build_json();
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    // #pragma omp parallel sections
    {
        // #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        // #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static void generate_proposals(const ncnn::Mat& pred, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;

    const int reg_max_1 = 16;
    const int num_class = pred.w - reg_max_1 * 4; // number of classes. 80 for COCO

    for (int y = 0; y < num_grid_y; y++)
    {
        for (int x = 0; x < num_grid_x; x++)
        {
            const ncnn::Mat pred_grid = pred.row_range(y * num_grid_x + x, 1);

            // find label with max score
            int label = -1;
            float score = -FLT_MAX;
            {
                const ncnn::Mat pred_score = pred_grid.range(reg_max_1 * 4, num_class);

                for (int k = 0; k < num_class; k++)
                {
                    float s = pred_score[k];
                    if (s > score)
                    {
                        label = k;
                        score = s;
                    }
                }

                score = sigmoid(score);
            }

            if (score >= prob_threshold)
            {
                ncnn::Mat pred_bbox = pred_grid.range(0, reg_max_1 * 4).reshape(reg_max_1, 4);

                {
                    ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                    ncnn::ParamDict pd;
                    pd.set(0, 1); // axis
                    pd.set(1, 1);
                    softmax->load_param(pd);

                    ncnn::Option opt;
                    opt.num_threads = 1;
                    opt.use_packing_layout = false;

                    softmax->create_pipeline(opt);

                    softmax->forward_inplace(pred_bbox, opt);

                    softmax->destroy_pipeline(opt);

                    delete softmax;
                }

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    const float* dis_after_sm = pred_bbox.row(k);
                    for (int l = 0; l < reg_max_1; l++)
                    {
                        dis += l * dis_after_sm[l];
                    }

                    pred_ltrb[k] = dis * stride;
                }

                float pb_cx = (x + 0.5f) * stride;
                float pb_cy = (y + 0.5f) * stride;

                float x0 = pb_cx - pred_ltrb[0];
                float y0 = pb_cy - pred_ltrb[1];
                float x1 = pb_cx + pred_ltrb[2];
                float y1 = pb_cy + pred_ltrb[3];

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob = score;

                objects.push_back(obj);
            }
        }
    }
}

static void generate_proposals(const ncnn::Mat& pred, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    int pred_row_offset = 0;
    for (size_t i = 0; i < strides.size(); i++)
    {
        const int stride = strides[i];

        const int num_grid_x = w / stride;
        const int num_grid_y = h / stride;
        const int num_grid = num_grid_x * num_grid_y;

        generate_proposals(pred.row_range(pred_row_offset, num_grid), stride, in_pad, prob_threshold, objects);
        pred_row_offset += num_grid;
    }
}

int YOLOv8_det::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    const int target_size = det_target_size;//640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    int img_w = rgb.cols;
    int img_h = rgb.rows;

    // ultralytics/cfg/models/v8/yolov8.yaml
    std::vector<int> strides(3);
    strides[0] = 8;
    strides[1] = 16;
    strides[2] = 32;
    const int max_stride = 32;

    // letterbox pad to multiple of max_stride
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);

    // letterbox pad to target_size rectangle
    int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov8.create_extractor();

    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    std::vector<Object> proposals;
    generate_proposals(out, strides, in_pad, prob_threshold, proposals);

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    // sort objects by area
    struct
    {
        bool operator()(const Object& a, const Object& b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);

    return 0;
}

int YOLOv8_det_coco::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    static cv::Scalar colors[] = {
        cv::Scalar( 67,  54, 244),
        cv::Scalar( 30,  99, 233),
        cv::Scalar( 39, 176, 156),
        cv::Scalar( 58, 183, 103),
        cv::Scalar( 81, 181,  63),
        cv::Scalar(150, 243,  33),
        cv::Scalar(169, 244,   3),
        cv::Scalar(188, 212,   0),
        cv::Scalar(150, 136,   0),
        cv::Scalar(175,  80,  76),
        cv::Scalar(195,  74, 139),
        cv::Scalar(220,  57, 205),
        cv::Scalar(235,  59, 255),
        cv::Scalar(193,   7, 255),
        cv::Scalar(152,   0, 255),
        cv::Scalar( 87,  34, 255),
        cv::Scalar( 85,  72, 121),
        cv::Scalar(158, 158, 158),
        cv::Scalar(125, 139,  96)
    };

//    for (size_t i = 0; i < objects.size(); i++)
//    {
//        const Object& obj = objects[i];
//
//        const cv::Scalar& color = colors[i % 19];
//
//        // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
//                // obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
//
//        cv::rectangle(rgb, obj.rect, color);
//
//        char text[256];
//        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
//
//        int baseLine = 0;
//        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//
//        int x = obj.rect.x;
//        int y = obj.rect.y - label_size.height - baseLine;
//        if (y < 0)
//            y = 0;
//        if (x + label_size.width > rgb.cols)
//            x = rgb.cols - label_size.width;
//
//        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
//                      cv::Scalar(255, 255, 255), -1);
//
//        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
//                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
//    }

    std::vector<STrack> output_stracks = bytetracker.update(objects);

    for (unsigned long i = 0; i < output_stracks.size(); i++)
    {
        std::vector<float> tlwh = output_stracks[i].tlwh;
        bool vertical = tlwh[2] / tlwh[3] > 1.6;
        if (tlwh[2] * tlwh[3] > 20 && !vertical)
        {
            cv::Scalar s = bytetracker.get_color(output_stracks[i].track_id);
            cv::putText(rgb, cv::format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5),
                        0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::rectangle(rgb, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
        }
    }

    return 0;
}

int YOLOv8_det_oiv7::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "Accordion", "Adhesive tape", "Aircraft", "Airplane", "Alarm clock", "Alpaca", "Ambulance", "Animal",
        "Ant", "Antelope", "Apple", "Armadillo", "Artichoke", "Auto part", "Axe", "Backpack", "Bagel",
        "Baked goods", "Balance beam", "Ball", "Balloon", "Banana", "Band-aid", "Banjo", "Barge", "Barrel",
        "Baseball bat", "Baseball glove", "Bat (Animal)", "Bathroom accessory", "Bathroom cabinet", "Bathtub",
        "Beaker", "Bear", "Bed", "Bee", "Beehive", "Beer", "Beetle", "Bell pepper", "Belt", "Bench", "Bicycle",
        "Bicycle helmet", "Bicycle wheel", "Bidet", "Billboard", "Billiard table", "Binoculars", "Bird",
        "Blender", "Blue jay", "Boat", "Bomb", "Book", "Bookcase", "Boot", "Bottle", "Bottle opener",
        "Bow and arrow", "Bowl", "Bowling equipment", "Box", "Boy", "Brassiere", "Bread", "Briefcase",
        "Broccoli", "Bronze sculpture", "Brown bear", "Building", "Bull", "Burrito", "Bus", "Bust", "Butterfly",
        "Cabbage", "Cabinetry", "Cake", "Cake stand", "Calculator", "Camel", "Camera", "Can opener", "Canary",
        "Candle", "Candy", "Cannon", "Canoe", "Cantaloupe", "Car", "Carnivore", "Carrot", "Cart", "Cassette deck",
        "Castle", "Cat", "Cat furniture", "Caterpillar", "Cattle", "Ceiling fan", "Cello", "Centipede",
        "Chainsaw", "Chair", "Cheese", "Cheetah", "Chest of drawers", "Chicken", "Chime", "Chisel", "Chopsticks",
        "Christmas tree", "Clock", "Closet", "Clothing", "Coat", "Cocktail", "Cocktail shaker", "Coconut",
        "Coffee", "Coffee cup", "Coffee table", "Coffeemaker", "Coin", "Common fig", "Common sunflower",
        "Computer keyboard", "Computer monitor", "Computer mouse", "Container", "Convenience store", "Cookie",
        "Cooking spray", "Corded phone", "Cosmetics", "Couch", "Countertop", "Cowboy hat", "Crab", "Cream",
        "Cricket ball", "Crocodile", "Croissant", "Crown", "Crutch", "Cucumber", "Cupboard", "Curtain",
        "Cutting board", "Dagger", "Dairy Product", "Deer", "Desk", "Dessert", "Diaper", "Dice", "Digital clock",
        "Dinosaur", "Dishwasher", "Dog", "Dog bed", "Doll", "Dolphin", "Door", "Door handle", "Doughnut",
        "Dragonfly", "Drawer", "Dress", "Drill (Tool)", "Drink", "Drinking straw", "Drum", "Duck", "Dumbbell",
        "Eagle", "Earrings", "Egg (Food)", "Elephant", "Envelope", "Eraser", "Face powder", "Facial tissue holder",
        "Falcon", "Fashion accessory", "Fast food", "Fax", "Fedora", "Filing cabinet", "Fire hydrant",
        "Fireplace", "Fish", "Flag", "Flashlight", "Flower", "Flowerpot", "Flute", "Flying disc", "Food",
        "Food processor", "Football", "Football helmet", "Footwear", "Fork", "Fountain", "Fox", "French fries",
        "French horn", "Frog", "Fruit", "Frying pan", "Furniture", "Garden Asparagus", "Gas stove", "Giraffe",
        "Girl", "Glasses", "Glove", "Goat", "Goggles", "Goldfish", "Golf ball", "Golf cart", "Gondola",
        "Goose", "Grape", "Grapefruit", "Grinder", "Guacamole", "Guitar", "Hair dryer", "Hair spray", "Hamburger",
        "Hammer", "Hamster", "Hand dryer", "Handbag", "Handgun", "Harbor seal", "Harmonica", "Harp",
        "Harpsichord", "Hat", "Headphones", "Heater", "Hedgehog", "Helicopter", "Helmet", "High heels",
        "Hiking equipment", "Hippopotamus", "Home appliance", "Honeycomb", "Horizontal bar", "Horse", "Hot dog",
        "House", "Houseplant", "Human arm", "Human beard", "Human body", "Human ear", "Human eye", "Human face",
        "Human foot", "Human hair", "Human hand", "Human head", "Human leg", "Human mouth", "Human nose",
        "Humidifier", "Ice cream", "Indoor rower", "Infant bed", "Insect", "Invertebrate", "Ipod", "Isopod",
        "Jacket", "Jacuzzi", "Jaguar (Animal)", "Jeans", "Jellyfish", "Jet ski", "Jug", "Juice", "Kangaroo",
        "Kettle", "Kitchen & dining room table", "Kitchen appliance", "Kitchen knife", "Kitchen utensil",
        "Kitchenware", "Kite", "Knife", "Koala", "Ladder", "Ladle", "Ladybug", "Lamp", "Land vehicle",
        "Lantern", "Laptop", "Lavender (Plant)", "Lemon", "Leopard", "Light bulb", "Light switch", "Lighthouse",
        "Lily", "Limousine", "Lion", "Lipstick", "Lizard", "Lobster", "Loveseat", "Luggage and bags", "Lynx",
        "Magpie", "Mammal", "Man", "Mango", "Maple", "Maracas", "Marine invertebrates", "Marine mammal",
        "Measuring cup", "Mechanical fan", "Medical equipment", "Microphone", "Microwave oven", "Milk",
        "Miniskirt", "Mirror", "Missile", "Mixer", "Mixing bowl", "Mobile phone", "Monkey", "Moths and butterflies",
        "Motorcycle", "Mouse", "Muffin", "Mug", "Mule", "Mushroom", "Musical instrument", "Musical keyboard",
        "Nail (Construction)", "Necklace", "Nightstand", "Oboe", "Office building", "Office supplies", "Orange",
        "Organ (Musical Instrument)", "Ostrich", "Otter", "Oven", "Owl", "Oyster", "Paddle", "Palm tree",
        "Pancake", "Panda", "Paper cutter", "Paper towel", "Parachute", "Parking meter", "Parrot", "Pasta",
        "Pastry", "Peach", "Pear", "Pen", "Pencil case", "Pencil sharpener", "Penguin", "Perfume", "Person",
        "Personal care", "Personal flotation device", "Piano", "Picnic basket", "Picture frame", "Pig",
        "Pillow", "Pineapple", "Pitcher (Container)", "Pizza", "Pizza cutter", "Plant", "Plastic bag", "Plate",
        "Platter", "Plumbing fixture", "Polar bear", "Pomegranate", "Popcorn", "Porch", "Porcupine", "Poster",
        "Potato", "Power plugs and sockets", "Pressure cooker", "Pretzel", "Printer", "Pumpkin", "Punching bag",
        "Rabbit", "Raccoon", "Racket", "Radish", "Ratchet (Device)", "Raven", "Rays and skates", "Red panda",
        "Refrigerator", "Remote control", "Reptile", "Rhinoceros", "Rifle", "Ring binder", "Rocket",
        "Roller skates", "Rose", "Rugby ball", "Ruler", "Salad", "Salt and pepper shakers", "Sandal",
        "Sandwich", "Saucer", "Saxophone", "Scale", "Scarf", "Scissors", "Scoreboard", "Scorpion",
        "Screwdriver", "Sculpture", "Sea lion", "Sea turtle", "Seafood", "Seahorse", "Seat belt", "Segway",
        "Serving tray", "Sewing machine", "Shark", "Sheep", "Shelf", "Shellfish", "Shirt", "Shorts",
        "Shotgun", "Shower", "Shrimp", "Sink", "Skateboard", "Ski", "Skirt", "Skull", "Skunk", "Skyscraper",
        "Slow cooker", "Snack", "Snail", "Snake", "Snowboard", "Snowman", "Snowmobile", "Snowplow",
        "Soap dispenser", "Sock", "Sofa bed", "Sombrero", "Sparrow", "Spatula", "Spice rack", "Spider",
        "Spoon", "Sports equipment", "Sports uniform", "Squash (Plant)", "Squid", "Squirrel", "Stairs",
        "Stapler", "Starfish", "Stationary bicycle", "Stethoscope", "Stool", "Stop sign", "Strawberry",
        "Street light", "Stretcher", "Studio couch", "Submarine", "Submarine sandwich", "Suit", "Suitcase",
        "Sun hat", "Sunglasses", "Surfboard", "Sushi", "Swan", "Swim cap", "Swimming pool", "Swimwear",
        "Sword", "Syringe", "Table", "Table tennis racket", "Tablet computer", "Tableware", "Taco", "Tank",
        "Tap", "Tart", "Taxi", "Tea", "Teapot", "Teddy bear", "Telephone", "Television", "Tennis ball",
        "Tennis racket", "Tent", "Tiara", "Tick", "Tie", "Tiger", "Tin can", "Tire", "Toaster", "Toilet",
        "Toilet paper", "Tomato", "Tool", "Toothbrush", "Torch", "Tortoise", "Towel", "Tower", "Toy",
        "Traffic light", "Traffic sign", "Train", "Training bench", "Treadmill", "Tree", "Tree house",
        "Tripod", "Trombone", "Trousers", "Truck", "Trumpet", "Turkey", "Turtle", "Umbrella", "Unicycle",
        "Van", "Vase", "Vegetable", "Vehicle", "Vehicle registration plate", "Violin", "Volleyball (Ball)",
        "Waffle", "Waffle iron", "Wall clock", "Wardrobe", "Washing machine", "Waste container", "Watch",
        "Watercraft", "Watermelon", "Weapon", "Whale", "Wheel", "Wheelchair", "Whisk", "Whiteboard", "Willow",
        "Window", "Window blind", "Wine", "Wine glass", "Wine rack", "Winter melon", "Wok", "Woman",
        "Wood-burning stove", "Woodpecker", "Worm", "Wrench", "Zebra", "Zucchini"
    };

    static cv::Scalar colors[] = {
        cv::Scalar( 67,  54, 244),
        cv::Scalar( 30,  99, 233),
        cv::Scalar( 39, 176, 156),
        cv::Scalar( 58, 183, 103),
        cv::Scalar( 81, 181,  63),
        cv::Scalar(150, 243,  33),
        cv::Scalar(169, 244,   3),
        cv::Scalar(188, 212,   0),
        cv::Scalar(150, 136,   0),
        cv::Scalar(175,  80,  76),
        cv::Scalar(195,  74, 139),
        cv::Scalar(220,  57, 205),
        cv::Scalar(235,  59, 255),
        cv::Scalar(193,   7, 255),
        cv::Scalar(152,   0, 255),
        cv::Scalar( 87,  34, 255),
        cv::Scalar( 85,  72, 121),
        cv::Scalar(158, 158, 158),
        cv::Scalar(125, 139,  96)
    };

    static const char* port_state_names[] = {"free", "occupied"};

    g_port_reconstructor.update(objects);

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        if (obj.label < 0 || obj.label > 1)
            continue;

        const cv::Scalar color = obj.label == 0 ? cv::Scalar(60, 210, 80) : cv::Scalar(50, 80, 240);
        cv::rectangle(rgb, obj.rect, color, 2);

        char text[64];
        snprintf(text, sizeof(text), "%s %.2f", port_state_names[obj.label], obj.prob);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseLine);
        int x = std::max(0, (int)obj.rect.x);
        int y = std::max(label_size.height + 2, (int)obj.rect.y);
        cv::rectangle(rgb, cv::Rect(cv::Point(x, y - label_size.height - 2),
                                    cv::Size(label_size.width + 4, label_size.height + baseLine + 4)),
                      color, -1);
        cv::putText(rgb, text, cv::Point(x + 2, y), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    g_port_reconstructor.draw(rgb);

    return 0;
}
