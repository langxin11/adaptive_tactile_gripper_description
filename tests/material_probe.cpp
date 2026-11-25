// 输出 urdfdom 完成材质名称解析后宿主与导轨的实际颜色。
#include <urdf_parser/urdf_parser.h>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    auto model = urdf::parseURDFFile(argv[1]);
    if (!model) return 2;
    for (const auto& [name, link] : model->links_) {
        for (const auto& visual : link->visual_array) {
            auto mesh = std::dynamic_pointer_cast<urdf::Mesh>(visual->geometry);
            bool rail = mesh && std::filesystem::path(mesh->filename).filename() == "mgn9_rail.stl";
            if (name != "host" && !rail) continue;
            if (!visual->material) return 3;
            const auto& c = visual->material->color;
            std::cout << (rail ? "rail" : "host") << ' ' << c.r << ' ' << c.g
                      << ' ' << c.b << ' ' << c.a << '\n';
        }
    }
}
