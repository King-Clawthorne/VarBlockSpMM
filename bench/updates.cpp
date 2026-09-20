#include "support.hpp"
#include <iostream>
namespace b = vbsr::bench;
int main() {
    try {
        auto h = vbsr::generate({256, 256, 4, 32, vbsr::Distribution::Bimodal, false, 31});
        auto other = h;
        for (auto& value : other.values)
            value *= 0.5f;
        auto changed = vbsr::generate({256, 256, 8, 32, vbsr::Distribution::Bimodal, false, 31});
        vbsr::Matrix target(h), first(h), second(other), structure(changed);
        vbsr::Plan plan(target, {32});
        auto input = b::make_input(b::panel_elements(h.scalar_cols(), 32));
        b::DeviceBuffer<float> x(input.size()), y(b::panel_elements(h.scalar_rows(), 32));
        x.upload(input);
        auto r1 = vbsr::cpu_reference(h, input, 32), r2 = vbsr::cpu_reference(other, input, 32);
        auto rs = vbsr::cpu_reference(changed, input, 32);
        int last = 0;
        b::print_environment(&h);
        b::print_timing_header();
        b::measure(
            "value_update",
            [&](int i) {
                last = i;
                target.update_values((i % 2 ? second : first).device_view().values,
                                     h.values.size());
                plan.execute(x.data(), y.data());
            },
            [&] { b::verify_output(last % 2 ? r2 : r1, y.data()); }, 20, 0);
        b::measure(
            "structure_replan",
            [&](int i) {
                last = i;
                vbsr::Plan rebuilt((i % 2 ? structure : first).device_view(), {32});
                rebuilt.execute(x.data(), y.data());
                b::check_cuda(cudaDeviceSynchronize());
            },
            [&] { b::verify_output(last % 2 ? rs : r1, y.data()); }, 20, 1);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
