# 1: Outline a situation in which one wants to return a std::optional<T> from a function instead of a T.
std::optional is used when a value of type T may or may not be present. An example could be:
```
    struct PersonInfo {
        std::string name{};
        unsigned int age{};
        std::optional<std::string> nickname{};
    };
```

# 2: When would one use a std::variant<T1, T2, T3> instead of std::tuple<T1, T2, T3>?
std::tuple is a compile time known collection of values of known types, in this case an object with three fields/values. std::variant is just one value, that could be of one of the specified types.
-> use tuple when all values coexist, use variant when a value is one thing and the active type may change.

# 3: What is the obvious difference between std::pair and std::complex on one side and std::tuple and std::array on the other side?
## pair vs. complex:
A pair can store values of two different types, whereas complex carries intent in the form of representing a complex number with two members (representing real and img part) of the same type.

## tuple vs. array:
A tuple is a heterogenous container, meaning it can store values of different types. An array is a homogenous container, with all elements being of the same type.

# 4: What is the obvious difference between std::pair and std::tuple on one side and std::array and std::complex on the other side?
## pair vs. tuple:
Both serve the same purpose, but a pair is limited to two members, whereas a tuple can be of arbitrary size.

## array vs. complex:
An array is collection of elements of the same type. Complex is a class template with two data members of the same type, representing a complex number.

# What is special about std::complex compared to the other class templates mentioned in the previous questions?
All other class templates are pure data containers that do not impose any mathematical requirements on the type stored. Complex carries semantics that only make sense on continuos value types.
