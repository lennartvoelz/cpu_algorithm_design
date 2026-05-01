#include <iostream>
#include <functional>
#include <complex>
#include <array>
#include <tuple>
#include <utility>
#include <variant>
#include <optional>
#include <algorithm>

using innerArray = std::array<std::complex<float>, 3>;
using innerPair = std::pair<std::complex<double>, int>;
using outerArray = std::array<innerPair, 2>;
using outerPair = std::pair<innerArray, bool>;
using Bar = std::tuple<outerArray, outerPair>;
using Foo = std::optional<std::variant<Bar, std::function<Bar(bool)>>>;

// 1.2.1
auto get_first_double(const Foo& foo) -> double {
    if (foo.has_value() && std::holds_alternative<Bar>(foo.value())) {
        return std::get<outerArray>(std::get<Bar>(foo.value()))[0].first.real();
    }
    return 42.0;
}

auto get_first_int(const Foo& foo) -> int {
    if (foo.has_value() && std::holds_alternative<Bar>(foo.value())) {
        return std::get<outerArray>(std::get<Bar>(foo.value()))[0].second;
    }
    return 42;
}

auto get_first_float(const Foo& foo) -> float {
    if (foo.has_value() && std::holds_alternative<Bar>(foo.value())) {
        return std::get<outerPair>(std::get<Bar>(foo.value())).first[0].real();
    }
    return 42.0f;
}

auto get_first_bool(const Foo& foo) -> bool {
    if (foo.has_value() && std::holds_alternative<Bar>(foo.value())) {
        return std::get<outerPair>(std::get<Bar>(foo.value())).second;
    }
    return true;
}

// 1.2.2
auto get_Bar(const Foo& foo) -> Bar {
    if (!foo.has_value()) {
        return Bar{};
    }
    const auto& val{ foo.value() };
    if (std::holds_alternative<Bar>(val)) {
        return std::get<Bar>(val);
    }
    return std::get<std::function<Bar(bool)>>(val)(true);
}

// 1.2.3
auto multiply(const Bar& bar_x, const Bar& bar_y) -> Bar {
    const auto& [array_x, pair_x] = bar_x;
    const auto& [array_y, pair_y] = bar_y;
    
    outerArray array_result{};
    auto array_multiply = [](const innerPair& a, const innerPair& b) -> innerPair {
        return { a.first * b.first, a.second * b.second };
    };
    std::transform(array_x.begin(), array_x.end(), array_y.begin(), array_result.begin(), array_multiply);

    const auto& [arr_a, bool_a] = pair_x;
    const auto& [arr_b, bool_b] = pair_y;
    innerArray inner_array_result{};
    std::transform(arr_a.begin(), arr_a.end(), arr_b.begin(), inner_array_result.begin(), std::multiplies<>{});

    return { array_result, { inner_array_result, bool_a && bool_b } };
}


int main() {}
