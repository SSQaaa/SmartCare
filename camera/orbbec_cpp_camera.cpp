#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "libobsensor/hpp/Context.hpp"
#include "libobsensor/hpp/Device.hpp"
#include "libobsensor/hpp/Error.hpp"
#include "libobsensor/hpp/Frame.hpp"
#include "libobsensor/hpp/Pipeline.hpp"
#include "libobsensor/hpp/StreamProfile.hpp"

namespace py = pybind11;

namespace {

class ScopedNativeOutputSilencer {
public:
    explicit ScopedNativeOutputSilencer(bool enabled) : enabled_(enabled) {
#ifdef _WIN32
        if(!enabled_) {
            return;
        }

        fflush(stdout);
        fflush(stderr);
        saved_stdout_ = _dup(_fileno(stdout));
        saved_stderr_ = _dup(_fileno(stderr));
        null_fd_ = _open("NUL", _O_WRONLY);

        if(saved_stdout_ >= 0 && saved_stderr_ >= 0 && null_fd_ >= 0) {
            _dup2(null_fd_, _fileno(stdout));
            _dup2(null_fd_, _fileno(stderr));
        }
#endif
    }

    ~ScopedNativeOutputSilencer() {
#ifdef _WIN32
        if(!enabled_) {
            return;
        }

        fflush(stdout);
        fflush(stderr);
        if(saved_stdout_ >= 0) {
            _dup2(saved_stdout_, _fileno(stdout));
            _close(saved_stdout_);
        }
        if(saved_stderr_ >= 0) {
            _dup2(saved_stderr_, _fileno(stderr));
            _close(saved_stderr_);
        }
        if(null_fd_ >= 0) {
            _close(null_fd_);
        }
#endif
    }

private:
    bool enabled_;
#ifdef _WIN32
    int saved_stdout_ = -1;
    int saved_stderr_ = -1;
    int null_fd_ = -1;
#endif
};

std::string format_name(OBFormat format) {
    switch(format) {
    case OB_FORMAT_BGR:
        return "BGR";
    case OB_FORMAT_RGB:
        return "RGB";
    case OB_FORMAT_Y16:
        return "Y16";
    case OB_FORMAT_MJPG:
        return "MJPG";
    case OB_FORMAT_YUYV:
        return "YUYV";
    case OB_FORMAT_UYVY:
        return "UYVY";
    case OB_FORMAT_I420:
        return "I420";
    default:
        return "UNKNOWN(" + std::to_string(static_cast<int>(format)) + ")";
    }
}

std::string ob_error_text(const std::string &stage, const ob::Error &error) {
    return "Orbbec C++ SDK failed while " + stage +
           ". function=" + std::string(error.getName() ? error.getName() : "") +
           ", args=" + std::string(error.getArgs() ? error.getArgs() : "") +
           ", message=" + std::string(error.getMessage() ? error.getMessage() : "") +
           ", type=" + std::to_string(static_cast<int>(error.getExceptionType()));
}

std::shared_ptr<ob::VideoStreamProfile> first_video_profile(
    const std::shared_ptr<ob::StreamProfileList> &profiles) {
    return std::const_pointer_cast<ob::StreamProfile>(profiles->getProfile(0))->as<ob::VideoStreamProfile>();
}

std::shared_ptr<ob::VideoStreamProfile> choose_color_profile(
    const std::shared_ptr<ob::StreamProfileList> &profiles,
    int width,
    int height,
    int fps) {
    const int query_width = width > 0 ? width : 0;
    const int query_height = height > 0 ? height : 0;
    const int query_fps = fps > 0 ? fps : 30;

    for(OBFormat format : { OB_FORMAT_BGR, OB_FORMAT_RGB }) {
        try {
            return profiles->getVideoStreamProfile(query_width, query_height, format, query_fps);
        }
        catch(const ob::Error &) {
        }
    }

    return first_video_profile(profiles);
}

std::shared_ptr<ob::VideoStreamProfile> choose_depth_profile(
    const std::shared_ptr<ob::StreamProfileList> &profiles,
    int width,
    int height,
    int fps) {
    const int query_width = width > 0 ? width : 0;
    const int query_height = height > 0 ? height : 0;
    const int query_fps = fps > 0 ? fps : 30;

    try {
        return profiles->getVideoStreamProfile(query_width, query_height, OB_FORMAT_Y16, query_fps);
    }
    catch(const ob::Error &) {
        return first_video_profile(profiles);
    }
}

py::array_t<uint8_t> color_frame_to_bgr(const std::shared_ptr<ob::ColorFrame> &frame) {
    const auto width = static_cast<py::ssize_t>(frame->width());
    const auto height = static_cast<py::ssize_t>(frame->height());
    const auto format = frame->format();
    const auto expected_size = static_cast<size_t>(width * height * 3);
    const auto data_size = static_cast<size_t>(frame->dataSize());
    const auto *src = static_cast<const uint8_t *>(frame->data());

    if(format != OB_FORMAT_BGR && format != OB_FORMAT_RGB) {
        throw std::runtime_error(
            "Unsupported color frame format from Orbbec C++ SDK: " + format_name(format) +
            ". Please select an RGB/BGR color stream profile.");
    }
    if(data_size < expected_size) {
        throw std::runtime_error("Color frame data is smaller than width*height*3.");
    }

    py::array_t<uint8_t> out(std::vector<py::ssize_t>{ height, width, static_cast<py::ssize_t>(3) });
    auto dst = static_cast<uint8_t *>(out.mutable_data());

    if(format == OB_FORMAT_BGR) {
        std::memcpy(dst, src, expected_size);
    }
    else {
        for(size_t i = 0; i < expected_size; i += 3) {
            dst[i + 0] = src[i + 2];
            dst[i + 1] = src[i + 1];
            dst[i + 2] = src[i + 0];
        }
    }

    return out;
}

py::array_t<uint16_t> depth_frame_to_mm(const std::shared_ptr<ob::DepthFrame> &frame) {
    const auto width = static_cast<py::ssize_t>(frame->width());
    const auto height = static_cast<py::ssize_t>(frame->height());
    const auto pixel_count = static_cast<size_t>(width * height);
    const auto data_size = static_cast<size_t>(frame->dataSize());

    if(data_size < pixel_count * sizeof(uint16_t)) {
        throw std::runtime_error("Depth frame data is smaller than width*height*sizeof(uint16_t).");
    }

    const auto *src = static_cast<const uint16_t *>(frame->data());
    const float scale = frame->getValueScale();
    py::array_t<uint16_t> out(std::vector<py::ssize_t>{ height, width });
    auto dst = static_cast<uint16_t *>(out.mutable_data());

    if(scale > 0.999f && scale < 1.001f) {
        std::memcpy(dst, src, pixel_count * sizeof(uint16_t));
    }
    else {
        for(size_t i = 0; i < pixel_count; ++i) {
            const float value = static_cast<float>(src[i]) * scale;
            dst[i] = static_cast<uint16_t>(std::clamp(value, 0.0f, 65535.0f));
        }
    }

    return out;
}

}  // namespace

class OrbbecCppCamera {
public:
    OrbbecCppCamera(int color_width = 0,
                    int color_height = 0,
                    int depth_width = 0,
                    int depth_height = 0,
                    int fps = 30,
                    bool align_depth_to_color = true,
                    bool mirror = false,
                    bool suppress_sdk_output = true)
        : color_width_(color_width),
          color_height_(color_height),
          depth_width_(depth_width),
          depth_height_(depth_height),
          fps_(fps),
          align_depth_to_color_(align_depth_to_color),
          mirror_(mirror),
          suppress_sdk_output_(suppress_sdk_output) {}

    ~OrbbecCppCamera() {
        stop();
    }

    void start() {
        if(started_) {
            return;
        }

        ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_NONE);
        ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_NONE);

        ScopedNativeOutputSilencer silence(suppress_sdk_output_);

        pipe_ = std::make_unique<ob::Pipeline>();
        config_ = std::make_shared<ob::Config>();

        auto color_profiles = pipe_->getStreamProfileList(OB_SENSOR_COLOR);
        color_profile_ = choose_color_profile(color_profiles, color_width_, color_height_, fps_);
        config_->enableStream(color_profile_);

        auto depth_profiles = pipe_->getStreamProfileList(OB_SENSOR_DEPTH);
        depth_profile_ = choose_depth_profile(depth_profiles, depth_width_, depth_height_, fps_);
        config_->enableStream(depth_profile_);

        if(align_depth_to_color_) {
            config_->setAlignMode(ALIGN_D2C_SW_MODE);
        }

        pipe_->start(config_);

        try {
            auto device = pipe_->getDevice();
            if(device->isPropertySupported(OB_PROP_COLOR_MIRROR_BOOL, OB_PERMISSION_WRITE)) {
                device->setBoolProperty(OB_PROP_COLOR_MIRROR_BOOL, mirror_);
            }
            if(device->isPropertySupported(OB_PROP_DEPTH_MIRROR_BOOL, OB_PERMISSION_WRITE)) {
                device->setBoolProperty(OB_PROP_DEPTH_MIRROR_BOOL, mirror_);
            }
        }
        catch(const ob::Error &) {
        }

        started_ = true;
    }

    void stop() {
        if(pipe_ && started_) {
            try {
                pipe_->stop();
            }
            catch(...) {
            }
        }
        started_ = false;
        config_.reset();
        color_profile_.reset();
        depth_profile_.reset();
        pipe_.reset();
    }

    py::object read(int timeout_ms = 100) {
        if(!started_ || !pipe_) {
            throw std::runtime_error("OrbbecCppCamera has not been started.");
        }

        std::shared_ptr<ob::FrameSet> frame_set;
        {
            py::gil_scoped_release release;
            ScopedNativeOutputSilencer silence(suppress_sdk_output_);
            frame_set = pipe_->waitForFrames(static_cast<uint32_t>(timeout_ms));
        }
        if(!frame_set) {
            return py::none();
        }

        auto color_frame = frame_set->colorFrame();
        auto depth_frame = frame_set->depthFrame();
        if(!color_frame || !depth_frame) {
            return py::none();
        }

        py::dict result;
        result["color"] = color_frame_to_bgr(color_frame);
        result["depth"] = depth_frame_to_mm(depth_frame);
        result["color_format"] = format_name(color_frame->format());
        result["depth_format"] = format_name(depth_frame->format());
        result["color_width"] = color_frame->width();
        result["color_height"] = color_frame->height();
        result["depth_width"] = depth_frame->width();
        result["depth_height"] = depth_frame->height();
        result["depth_scale"] = depth_frame->getValueScale();
        return std::move(result);
    }

    py::dict info() const {
        py::dict result;
        result["started"] = started_;
        if(color_profile_) {
            result["color_width"] = color_profile_->width();
            result["color_height"] = color_profile_->height();
            result["color_fps"] = color_profile_->fps();
            result["color_format"] = format_name(color_profile_->format());
        }
        if(depth_profile_) {
            result["depth_width"] = depth_profile_->width();
            result["depth_height"] = depth_profile_->height();
            result["depth_fps"] = depth_profile_->fps();
            result["depth_format"] = format_name(depth_profile_->format());
        }
        return result;
    }

private:
    int color_width_;
    int color_height_;
    int depth_width_;
    int depth_height_;
    int fps_;
    bool align_depth_to_color_;
    bool mirror_;
    bool suppress_sdk_output_;
    bool started_ = false;

    std::unique_ptr<ob::Pipeline> pipe_;
    std::shared_ptr<ob::Config> config_;
    std::shared_ptr<ob::VideoStreamProfile> color_profile_;
    std::shared_ptr<ob::VideoStreamProfile> depth_profile_;
};

PYBIND11_MODULE(orbbec_cpp_camera, m) {
    m.doc() = "Minimal Orbbec C++ SDK camera wrapper for smart_care.";

    py::class_<OrbbecCppCamera>(m, "OrbbecCppCamera")
        .def(py::init<int, int, int, int, int, bool, bool, bool>(),
             py::arg("color_width") = 0,
             py::arg("color_height") = 0,
             py::arg("depth_width") = 0,
             py::arg("depth_height") = 0,
             py::arg("fps") = 30,
             py::arg("align_depth_to_color") = true,
             py::arg("mirror") = false,
             py::arg("suppress_sdk_output") = true)
        .def("start", &OrbbecCppCamera::start)
        .def("stop", &OrbbecCppCamera::stop)
        .def("read", &OrbbecCppCamera::read, py::arg("timeout_ms") = 100)
        .def("info", &OrbbecCppCamera::info);
}
