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

package com.tencent.yolov8ncnn;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.webkit.JavascriptInterface;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

public class MainActivity extends Activity implements SurfaceHolder.Callback
{
    public static final int REQUEST_CAMERA = 100;

    private YOLOv8Ncnn yolov8ncnn = new YOLOv8Ncnn();
    private int facing = 0;

    private int current_task = 0;
    private int current_model = 0;
    private int current_cpugpu = 0;

    private SurfaceView cameraView;
    private WebView webView;
    private Handler mainHandler;

    public class AndroidBridge
    {
        @JavascriptInterface
        public void loadModel(int task, int model, int cpugpu)
        {
            Log.d("MainActivity", "AndroidBridge.loadModel task=" + task + " model=" + model + " cpugpu=" + cpugpu);
            current_task = task;
            current_model = model;
            current_cpugpu = cpugpu;
            reload();
        }

        @JavascriptInterface
        public void switchCamera()
        {
            Log.d("MainActivity", "AndroidBridge.switchCamera");
            int new_facing = 1 - facing;
            yolov8ncnn.closeCamera();
            yolov8ncnn.openCamera(new_facing);
            facing = new_facing;
        }

        @JavascriptInterface
        public String getInitialState()
        {
            return "{\"task\":" + current_task + ",\"model\":" + current_model + ",\"cpugpu\":" + current_cpugpu + ",\"facing\":" + facing + "}";
        }
    }

    public void onDetectionResult(final String json)
    {
        mainHandler.post(new Runnable() {
            @Override
            public void run()
            {
                webView.evaluateJavascript("window.onDetectionResult && window.onDetectionResult(" + json + ")", null);
            }
        });
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        mainHandler = new Handler(Looper.getMainLooper());

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview);
        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);

        webView = (WebView) findViewById(R.id.webview_overlay);
        webView.setBackgroundColor(Color.TRANSPARENT);
        webView.setLayerType(WebView.LAYER_TYPE_HARDWARE, null);

        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setAllowFileAccess(true);
        settings.setAllowFileAccessFromFileURLs(true);
        settings.setAllowUniversalAccessFromFileURLs(true);
        settings.setDomStorageEnabled(true);

        webView.addJavascriptInterface(new AndroidBridge(), "Android");
        webView.setWebViewClient(new WebViewClient());
        webView.loadUrl("file:///android_asset/react_ui/index.html");

        reload();
    }

    private void reload()
    {
        boolean ret_init = yolov8ncnn.loadModel(getAssets(), current_task, current_model, current_cpugpu);
        if (!ret_init)
        {
            Log.e("MainActivity", "yolov8ncnn loadModel failed");
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height)
    {
        yolov8ncnn.setOutputWindow(holder.getSurface());
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder)
    {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder)
    {
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED)
        {
            requestPermissions(new String[] {Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }

        yolov8ncnn.openCamera(facing);
        yolov8ncnn.setCallbackObject(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();

        yolov8ncnn.closeCamera();
    }
}
