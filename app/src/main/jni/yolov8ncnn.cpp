// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "yolov8.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

// Global JVM and callback references
static JavaVM* g_jvm = 0;
static jobject g_callback_obj = 0;
static jmethodID g_callback_method = 0;

static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static float compute_avg_fps()
{
    static double t0 = 0.f;
    static float fps_history[10] = {0.f};

    double t1 = ncnn::get_current_time();
    if (t0 == 0.f)
    {
        t0 = t1;
        return 0.f;
    }

    float fps = 1000.f / (t1 - t0);
    t0 = t1;

    for (int i = 9; i >= 1; i--)
    {
        fps_history[i] = fps_history[i - 1];
    }
    fps_history[0] = fps;

    if (fps_history[9] == 0.f)
    {
        return 0.f;
    }

    float avg_fps = 0.f;
    for (int i = 0; i < 10; i++)
    {
        avg_fps += fps_history[i];
    }
    avg_fps /= 10.f;

    return avg_fps;
}

static int draw_fps(cv::Mat& rgb, float avg_fps)
{
    if (avg_fps == 0.f)
        return 0;

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

static std::string build_detection_json(const std::vector<Object>& objects, float fps)
{
    char buf[4096];
    int pos = 0;

    pos += snprintf(buf + pos, sizeof(buf) - pos, "{\"fps\":%.2f,\"count\":%d,\"objects\":[", fps, (int)objects.size());

    for (int i = 0; i < (int)objects.size(); i++)
    {
        if (i > 0)
            pos += snprintf(buf + pos, sizeof(buf) - pos, ",");

        const Object& obj = objects[i];
        pos += snprintf(buf + pos, sizeof(buf) - pos,
            "{\"label\":%d,\"prob\":%.4f,\"x\":%.1f,\"y\":%.1f,\"w\":%.1f,\"h\":%.1f}",
            obj.label, obj.prob,
            obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
    }

    pos += snprintf(buf + pos, sizeof(buf) - pos, "]}");

    return std::string(buf);
}

static void fire_java_callback(const std::string& json)
{
    if (!g_jvm || !g_callback_obj || !g_callback_method)
        return;

    JNIEnv* env = 0;
    bool attached = false;

    int ret = g_jvm->GetEnv((void**)&env, JNI_VERSION_1_4);
    if (ret == JNI_EDETACHED)
    {
        g_jvm->AttachCurrentThread(&env, 0);
        attached = true;
    }
    else if (ret != JNI_OK)
    {
        return;
    }

    jstring jstr = env->NewStringUTF(json.c_str());
    env->CallVoidMethod(g_callback_obj, g_callback_method, jstr);
    env->DeleteLocalRef(jstr);

    if (env->ExceptionCheck())
    {
        env->ExceptionClear();
    }

    if (attached)
    {
        g_jvm->DetachCurrentThread();
    }
}

static YOLOv8* g_yolov8 = 0;
static ncnn::Mutex lock;

class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    std::vector<Object> objects;

    // yolov8
    {
        ncnn::MutexLockGuard g(lock);

        if (g_yolov8)
        {
            g_yolov8->detect(rgb, objects);

            g_yolov8->draw(rgb, objects);
        }
        else
        {
            draw_unsupported(rgb);
        }
    }

    float avg_fps = compute_avg_fps();
    draw_fps(rgb, avg_fps);

    // Fire callback to Java with detection results
    if (avg_fps > 0.f)
    {
        std::string json = build_detection_json(objects, avg_fps);
        fire_java_callback(json);
    }
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_jvm = vm;

    g_camera = new MyNdkCamera;

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_yolov8;
        g_yolov8 = 0;
    }

    if (g_callback_obj)
    {
        JNIEnv* env = 0;
        if (vm->GetEnv((void**)&env, JNI_VERSION_1_4) == JNI_OK)
        {
            env->DeleteGlobalRef(g_callback_obj);
        }
        g_callback_obj = 0;
    }

    ncnn::destroy_gpu_instance();

    delete g_camera;
    g_camera = 0;
}

// public native void setCallbackObject(Object callbackObj);
JNIEXPORT void JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_setCallbackObject(JNIEnv* env, jobject thiz, jobject callbackObj)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setCallbackObject %p", callbackObj);

    // Release previous global reference
    if (g_callback_obj)
    {
        env->DeleteGlobalRef(g_callback_obj);
        g_callback_obj = 0;
    }

    if (callbackObj)
    {
        g_callback_obj = env->NewGlobalRef(callbackObj);

        jclass cls = env->GetObjectClass(callbackObj);
        g_callback_method = env->GetMethodID(cls, "onDetectionResult", "(Ljava/lang/String;)V");
        env->DeleteLocalRef(cls);

        if (!g_callback_method)
        {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "setCallbackObject: onDetectionResult method not found");
        }
    }
}

// public native boolean loadModel(AssetManager mgr, int taskid, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint taskid, jint modelid, jint cpugpu)
{
    if (taskid < 0 || taskid > 5 || modelid < 0 || modelid > 8 || cpugpu < 0 || cpugpu > 2)
    {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    const char* tasknames[6] =
    {
        "",
        "_oiv7",
        "_seg",
        "_pose",
        "_cls",
        "_obb"
    };

    const char* modeltypes[9] =
    {
        "n",
        "s",
        "m",
        "n",
        "s",
        "m",
        "n",
        "s",
        "m"
    };

    std::string parampath = std::string("yolov8") + modeltypes[(int)modelid] + tasknames[(int)taskid] + ".ncnn.param";
    std::string modelpath = std::string("yolov8") + modeltypes[(int)modelid] + tasknames[(int)taskid] + ".ncnn.bin";
    bool use_gpu = (int)cpugpu == 1;
    bool use_turnip = (int)cpugpu == 2;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        {
            static int old_taskid = 0;
            static int old_modelid = 0;
            static int old_cpugpu = 0;
            if (taskid != old_taskid || (modelid % 3) != old_modelid || cpugpu != old_cpugpu)
            {
                // taskid or model or cpugpu changed
                delete g_yolov8;
                g_yolov8 = 0;
            }
            old_taskid = taskid;
            old_modelid = modelid % 3;
            old_cpugpu = cpugpu;

            ncnn::destroy_gpu_instance();

            if (use_turnip)
            {
                ncnn::create_gpu_instance("libvulkan_freedreno.so");
            }
            else if (use_gpu)
            {
                ncnn::create_gpu_instance();
            }

            if (!g_yolov8)
            {
                if (taskid == 0) g_yolov8 = new YOLOv8_det_coco;
                if (taskid == 1) g_yolov8 = new YOLOv8_det_oiv7;
                if (taskid == 2) g_yolov8 = new YOLOv8_seg;
                if (taskid == 3) g_yolov8 = new YOLOv8_pose;
                if (taskid == 4) g_yolov8 = new YOLOv8_cls;
                if (taskid == 5) g_yolov8 = new YOLOv8_obb;

                g_yolov8->load(mgr, parampath.c_str(), modelpath.c_str(), use_gpu || use_turnip);
            }
            int target_size = 320;
            if ((int)modelid >= 3)
                target_size = 480;
            if ((int)modelid >= 6)
                target_size = 640;
            g_yolov8->set_det_target_size(target_size);
        }
    }

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int)facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

}
