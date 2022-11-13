#ifdef LOCAL
    #define _GLIBCXX_DEBUG
    #define _GLIBCXX_DEBUG_PEDANTIC
#endif
//#define NOGNU
#ifndef LOCAL
    #pragma GCC optimize("Ofast")
    #pragma GCC optimize("no-stack-protector")
#endif
#pragma GCC target("avx2")
// #pragma GCC target("sse4")
// #pragma GCC target("sse2")
#pragma GCC target("bmi2")
#pragma GCC target("popcnt")
#include <bits/stdc++.h>
#include <immintrin.h>
#ifdef IACA
#include <iacaMarks.h>
#else
#define IACA_START
#define IACA_END
#endif
// #define FILENAME "delivery"
#ifndef NOGNU
    #include <ext/rope>
    #include <ext/pb_ds/assoc_container.hpp>
#endif // NOGNU
#define lambda(body, ...) [&][[gnu::always_inline]](__VA_ARGS__) { return body; }
#define vlambda(body, ...) lambda(body, __VA_ARGS__ __VA_OPT__(,) auto&&...)
#define trans(...) views::transform(lambda(__VA_ARGS__))
#define vtrans(...) views::transform(vlambda(__VA_ARGS__))
#define all(x) (x).begin(), (x).end()
#define F first
#define S second
#define eb emplace_back
#define pb push_back
#define unires(x) x.resize(unique(all(x)) - x.begin())
#define ll long long
#define yesno(x) ((x) ? "YES" : "NO")
#ifdef LOCAL
    #define assume assert
#else
    #ifdef NOGNU
        #define assume(...)
    #else 
        #define assume(...) if (!(__VA_ARGS__)) __builtin_unreachable()
    #endif
#endif

#define XCAT(x, y) x##y
#define CAT(x, y) XCAT(x, y)

#define MAKE_INT_TYPE_SHORTCUT(bits) \
using CAT(i, bits) = CAT(int, CAT(bits, _t)); \
using CAT(u, bits) = CAT(uint, CAT(bits, _t));

MAKE_INT_TYPE_SHORTCUT(8)
MAKE_INT_TYPE_SHORTCUT(16)
MAKE_INT_TYPE_SHORTCUT(32)
MAKE_INT_TYPE_SHORTCUT(64)

#undef MAKE_INT_TYPE_SHORTCUT

#ifndef NOGNU
    
using u128 = unsigned __int128;
using i128 = __int128;

using i_max = i128;
using u_max = u128;

#else

using i_max = i64;
using u_max = u64;

#endif

using ld = long double;
using pii = std::pair<int, int>;
using pll = std::pair<ll, ll>;
using puu = std::pair<u32, u32>;
using puu64 = std::pair<u64, u64>;

const unsigned ll M1 = 4294967291, M2 = 4294967279, M = 998244353;

#ifndef M_PI
    const ld M_PI = acos(-1);
#endif // M_PI

using namespace std;
namespace rng = std::ranges;

#ifndef NOGNU
    using namespace __gnu_cxx;
    using namespace __gnu_pbds;

    template<class K, class T, class Cmp = less<K>>
    using ordered_map = tree<K, T, Cmp, rb_tree_tag, tree_order_statistics_node_update>;

    template<class T, class Cmp = less<T>>
    using ordered_set = ordered_map<T, null_type, Cmp>;
#endif

void run();

template<bool neg>
struct Inf {
    constexpr Inf() {}
    template<class T>
    [[nodiscard, gnu::pure]] constexpr operator T() const requires(std::integral<T> || std::floating_point<T>) {
        // No infinity for fast-math
        if constexpr (neg) {
            static_assert(is_signed_v<T>);
            return numeric_limits<T>::min() / 2;
        } else {
            return numeric_limits<T>::max() / 2;
        }
    }
    template<class T>
    [[nodiscard, gnu::pure]] constexpr auto operator<=>(const T &x) const {
        return (T) *this <=> x;
    }
    template<class T>
    [[nodiscard, gnu::pure]] constexpr auto operator==(const T &x) const {
        return (T) *this == x;
    }
    Inf &operator=(const Inf&) = delete;
    Inf &operator=(Inf&&) = delete;
    Inf(const Inf&) = delete;
    Inf(Inf&&) = delete;
    [[nodiscard]] Inf<!neg> operator-() const {
        return {};
    }
};

Inf<0> inf;

template<class T1, class T2>
[[gnu::always_inline]] inline bool mini(T1 &&a, T2 &&b) {
    if (a > b) {
        a = b;
        return 1;
    }
    return 0;
}

template<class T1, class T2>
[[gnu::always_inline]] inline bool maxi(T1 &&a, T2 &&b) {
    if (a < b) {
        a = b;
        return 1;
    }
    return 0;
}

mt19937 rnd(0);

signed main() {
    #if defined FILENAME && !defined STDIO && !defined LOCAL
        freopen(FILENAME".in", "r", stdin);
        freopen(FILENAME".out", "w", stdout);
    #endif // FILENAME
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    #ifdef LOCAL
    _mm_setcsr(_mm_getcsr() & ~(0x0080 | 0x0200));
    #else
    rnd.seed(random_device{}());
    #endif
    cout << setprecision(11) << fixed;
    #ifdef PRINTTIME
        auto start = clock();
    #endif
    run();
    #ifdef PRINTTIME
        cout << "\ntime = " << (double)(clock() - start) / CLOCKS_PER_SEC << "\n";
    #endif
    return 0;
}

#define rand rnd

template<class T>
struct is_tuple_like {
 private:
    static void detect(...);
    template<class U>
    static enable_if_t<(bool) tuple_size<U>::value, int> detect(const U&);
 public:
    static constexpr bool value = !is_same_v<decltype(detect(declval<remove_reference_t<T>>())), void>;
};

template<class T>
inline constexpr bool is_tuple_like_v = is_tuple_like<T>::value;

template<class T>
concept tuple_like = is_tuple_like_v<T>;

template<class Stream>
struct StreamWrapper {
    Stream &stream;
    [[gnu::always_inline]] StreamWrapper(Stream &stream): stream(stream) {}
    [[gnu::always_inline]] operator Stream&() const {
        return stream;
    }
};

namespace std {

template<std::ranges::range T>
[[gnu::always_inline]] inline ostream &operator<<(StreamWrapper<ostream> wout, const T &arr) {
    auto &out = wout.stream;
    bool first = 1;
    for (auto &&x : arr) {
        if (!first) {
            if constexpr (
                (std::ranges::range<decltype(x)> &&
                    !is_same_v<decltype(x), string>)
                || is_tuple_like_v<decltype(x)>) {
                out << '\n';
            } else {
                out << ' ';
            }
        } else {
            first = 0;
        }
        out << x;
    }
    return out;
}

template<std::ranges::range T>
[[gnu::always_inline]] inline istream &operator>>(StreamWrapper<istream> win, T &arr) {
    auto &in = win.stream;
    for (auto &x : arr)
        in >> x;
    return in;
}

};

template<class T, size_t pos>
[[gnu::always_inline]] inline void read_tuple(istream &in, T &x) {
    if constexpr (pos == tuple_size_v<T>)
        return;
    else {
        in >> get<pos>(x);
        read_tuple<T, pos + 1>(in, x);
    }
}

template<class T, size_t pos>
[[gnu::always_inline]] inline void write_tuple(ostream &out, const T &x) {
    if constexpr (pos == tuple_size_v<T>)
        return;
    else {
        if constexpr (pos != 0)
            out << " ";
        out << get<pos>(x);
        write_tuple<T, pos + 1>(out, x);
    }
}

namespace std {

template<tuple_like T>
requires(!std::ranges::range<T>)
[[gnu::always_inline]] inline istream &operator>>(StreamWrapper<istream> in, T &x) {
    read_tuple<T, 0>(in, x);
    return in;
}

template<tuple_like T>
requires(!std::ranges::range<T>)
[[gnu::always_inline]] inline ostream &operator<<(StreamWrapper<ostream> out, const T &x) {
    write_tuple<T, 0>(out, x);
    return out;
}

};

template<std::integral Int>
[[gnu::always_inline]] inline auto range(Int n) {
    return views::iota(static_cast<Int>(0), n);
}

template<class T, class... Ts, class... Args>
[[gnu::always_inline]] inline auto input(Args&&... args) {
    if constexpr (sizeof...(Ts) == 0) {
        T x(forward<Args>(args)...);
        cin >> x;
        return x;
    } else {
        return input<tuple<T, Ts...>>(forward<Args>(args)...);
    }
}

namespace segtree {

#ifdef LOCAL
#pragma GCC diagnostic ignored "-Winaccessible-base"
#endif

struct DoNothingFunc {
    template<class... Args>
    [[gnu::always_inline]] void operator()(Args&&... args) const {}
};

struct AssignFunc {
    template<class T, class U, class... Args>
    [[gnu::always_inline]] void operator()(T &x, U &&y, Args&&...) const {
        x = forward<U>(y);
    }
};

template<class F, class Res, class... Args>
[[gnu::always_inline]] inline
typename enable_if<
    is_void_v<decltype(declval<F>()(declval<Res&>(), declval<Args>()...))>,
    void
>::type
assign_or_call(F &f, Res &res, Args&&... args) {
    f(res, forward<Args>(args)...);
}

template<class F, class Res, class... Args>
[[gnu::always_inline]] inline 
typename enable_if<
    !is_void_v<decltype(declval<F>()(declval<Args>()...))>,
    void
>::type
assign_or_call(F &f, Res &res, Args&&... args) {
    res = f(forward<Args>(args)...);
}

template<int priority, class calc_t, class recalc_t, class join_t, class Self>
struct PointUpdatePolicyType {
    static constexpr int update_priority = priority;
    static constexpr int init_priority = numeric_limits<int>::min();
    [[no_unique_address]] calc_t calc;
    [[no_unique_address]] recalc_t recalc;
    [[no_unique_address]] join_t join;
    template<class F, class Policy>
    [[gnu::always_inline]] static auto select(Policy policy) {
        if constexpr (is_default_constructible_v<F>)
            return F();
        else
            return policy.template get<F>();
    }
    template<class Policy>
    PointUpdatePolicyType(Policy policy)
        : calc(select<calc_t>(policy))
        , recalc(select<recalc_t>(policy))
        , join(select<join_t>(policy)) {}
    PointUpdatePolicyType(const PointUpdatePolicyType &) = default;
    PointUpdatePolicyType(PointUpdatePolicyType &&) = default;
    template<class... Args>
    void upd_impl(size_t p, size_t v, size_t l, size_t r, Args&&... args) {
        auto self = static_cast<Self *>(this);
        self->push(v, l, r);
        assign_or_call(recalc, self->get_val(self->a[v]), forward<Args>(args)..., l, r);
        if (r - l == 1) {
            assign_or_call(calc, self->get_val(self->a[v]), forward<Args>(args)..., l, r);
            return;
        }
        size_t c = (l + r) / 2;
        size_t left = self->get_left(v, l, r);
        size_t right = self->get_right(v, l, r);
        if (p < c)
            upd_impl(p, left, l, c, forward<Args>(args)...);
        else
            upd_impl(p, right, c, r, forward<Args>(args)...);
        assign_or_call(
            join,
            self->get_val(self->a[v]),
            self->get_val(self->a[left]),
            self->get_val(self->a[right]),
            forward<Args>(args)...,
            l, r);
    }
    template<class... Args>
    [[gnu::always_inline]] void upd(size_t p, Args&&... args) {
        auto self = static_cast<Self *>(this);
        upd_impl(p, self->get_root(), 0, self->n, forward<Args>(args)...);
    }
    [[gnu::always_inline]] void push(size_t, size_t, size_t) {}
};

template<class recalc_t, int priority = 0>
struct PathUpdatePolicy {
    [[no_unique_address]] recalc_t recalc;
    template<class T>
    [[gnu::always_inline]] T get() {
        return recalc;
    }
    PathUpdatePolicy(recalc_t recalc): recalc(recalc) {}
    template<class Self>
    using type = PointUpdatePolicyType<
        priority,
        DoNothingFunc,
        recalc_t,
        DoNothingFunc,
        Self
    >;
};

template<class join_t, class convert_t, int priority = 0>
struct JoinUpdatePolicy {
    [[no_unique_address]] join_t join;
    [[no_unique_address]] convert_t convert;
    template<class T>
    [[gnu::always_inline]] T get() {
        if constexpr (is_same_v<T, join_t>)
            return T(join);
        else
            return T(convert);
    }
    JoinUpdatePolicy(join_t join, convert_t convert): join(join), convert(convert) {}
    template<class Self>
    using type = PointUpdatePolicyType<
        priority,
        convert_t,
        DoNothingFunc,
        join_t,
        Self
    >;
};

template<int priority, class gen_t, class Self>
struct SegTreeInitType {
    static constexpr int init_priority = priority;
    static constexpr int update_priority = numeric_limits<int>::min();
    [[gnu::always_inline]] void push(size_t, size_t, size_t) {}
    [[no_unique_address]] gen_t gen;
    template<class Policy>
    SegTreeInitType(Policy policy): gen(policy.template get<gen_t>()) {}
    void build(size_t v, size_t l, size_t r) {
        auto self = static_cast<Self *>(this); 
        if (r - l == 1) {
            if constexpr (!is_invocable_v<gen_t, size_t, size_t>)
                self->get_val(self->a[v]) = gen(l);
        } else {
            size_t c = (l + r) / 2;
            auto left = self->get_left(v, l, r);
            auto right = self->get_right(v, l, r);
            build(left, l, c);
            build(right, c, r);
            if constexpr (!is_invocable_v<gen_t, size_t, size_t>)
                self->join(
                    self->get_val(self->a[v]),
                    self->get_val(self->a[left]),
                    self->get_val(self->a[right]));
        }
        if constexpr (is_invocable_v<gen_t, size_t, size_t>) {
            self->get_val(self->a[v]) = init(l, r);
        }
    }
    [[gnu::always_inline]] void init() {
        auto self = static_cast<Self *>(this);
        build(self->get_root(), 0, self->n);
    }
    [[gnu::always_inline]] auto init(size_t l, size_t r) {
        if constexpr (is_invocable_v<gen_t, size_t, size_t>)
            return gen(l, r);
        else
            return gen();
    }
};

template<class gen_t>
struct InitPolicy {
    [[no_unique_address]] gen_t gen;
    InitPolicy(gen_t gen): gen(gen) {}
    template<class T>
    [[gnu::always_inline]] T get() {
        return gen;
    }
    template<class Self>
    using type = SegTreeInitType<0, gen_t, Self>;
};

struct IdentityFunc {
    template<class T, class... Args>
    [[gnu::always_inline, gnu::const]] const T &operator()(const T &x, Args&&...) {
        return x;
    }
};

template<class join_t, class identity_t, class convert_t = IdentityFunc>
struct GetPolicy {
    [[no_unique_address]] join_t join;
    [[no_unique_address]] identity_t identity;
    [[no_unique_address]] convert_t convert;
    template<class T>
    [[gnu::always_inline]] T get() {
        if constexpr (is_same_v<T, join_t>)
            return join;
        else if constexpr (is_same_v<T, identity_t>)
            return identity;
    }
    [[gnu::always_inline]]
    GetPolicy(join_t join, identity_t identity, convert_t convert = convert_t()):
        join(join),
        identity(identity),
        convert(convert) {}
    template<class Self>
    struct type
        : public PointUpdatePolicyType<
            -1,
            AssignFunc,
            DoNothingFunc,
            join_t,
            Self
        >
        , public SegTreeInitType<-1, identity_t, Self>
    {
        static constexpr int init_priority = -1;
        static constexpr int update_priority = -1;
        [[no_unique_address]] GetPolicy policy;
        type(GetPolicy policy)
            : PointUpdatePolicyType<
                -1,
                AssignFunc,
                DoNothingFunc,
                join_t,
                Self
            >(policy)
            , SegTreeInitType<-1, identity_t, Self>(policy)
            , policy(policy) {}
        template<class... Args>
        auto get_impl(size_t ql, size_t qr, size_t v, size_t l, size_t r, Args&&... args) {
            auto self = static_cast<Self *>(this);
            if (qr <= l || r <= ql) {
                return static_cast<
                        decltype(policy.convert(self->get_val(self->a[v]), forward<Args>(args)..., l, r))
                    >(policy.identity(forward<Args>(args)..., l, r));
            }
            self->push(v, l, r);
            if (ql <= l && r <= qr)
                return policy.convert(self->get_val(self->a[v]), forward<Args>(args)..., l, r);
            size_t c = (l + r) / 2;
            return policy.join(
                get_impl(ql, qr, self->get_left(v, l, r), l, c, forward<Args>(args)...),
                get_impl(ql, qr, self->get_right(v, l, r), c, r, forward<Args>(args)...),
                forward<Args>(args)..., l, r);
        }
        template<class... Args>
        [[gnu::always_inline]] auto get(size_t ql, size_t qr, Args&&... args) {
            auto self = static_cast<Self *>(this);
            return get_impl(ql, qr, self->get_root(), 0, self->n, forward<Args>(args)...);
        }
        [[gnu::always_inline]] void push(size_t, size_t, size_t) {}
    };
};

template<class push_t, class balance_t, class calc_t>
struct MassUpdatePolicy {
    [[no_unique_address]] push_t push;
    [[no_unique_address]] balance_t balance;
    [[no_unique_address]] calc_t calc;
    MassUpdatePolicy(push_t push, balance_t balance, calc_t calc):
        push(push),
        balance(balance),
        calc(calc) {}
    template<class Self>
    struct type {
        static const int init_priority = numeric_limits<int>::min();
        static const int update_priority = numeric_limits<int>::min();
        [[no_unique_address]] MassUpdatePolicy policy;
        type(MassUpdatePolicy policy): policy(policy) {}
        [[gnu::always_inline]] void push(size_t v, size_t l, size_t r) {
            auto self = static_cast<Self *>(this);
            if (r - l != 1) {
                auto left = self->get_left(v, l, r);
                auto right = self->get_right(v, l, r);
                size_t c = (l + r) / 2;
                policy.push(self->get_val(self->a[left]), l, c, self->get_val(self->a[v]), l, r);
                policy.push(self->get_val(self->a[right]), c, r, self->get_val(self->a[v]), l, r);
            }
            policy.balance(self->get_val(self->a[v]), l, r);
        }
        template<class... Args>
        void mass_upd_impl(size_t ql, size_t qr, size_t v, size_t l, size_t r, Args&&... args) {
            auto self = static_cast<Self *>(this);
            if (qr <= l || r <= ql)
                return;
            self->push(v, l, r);
            if (ql <= l && r <= qr) {
                policy.calc(self->get_val(self->a[v]), forward<Args>(args)..., l, r);
                return;
            }
            size_t c = (l + r) / 2;
            mass_upd_impl(ql, qr, self->get_left(v, l, r), l, c, forward<Args>(args)...);
            mass_upd_impl(ql, qr, self->get_right(v, l, r), c, r, forward<Args>(args)...);
            auto left = self->get_left(v, l, r);
            auto right = self->get_right(v, l, r);
            self->join(self->get_val(self->a[v]), self->get_val(self->a[left]), self->get_val(self->a[right]));
        }
        template<class... Args>
        [[gnu::always_inline]] void mass_upd(size_t ql, size_t qr, Args&&... args) {
            auto self = static_cast<Self *>(this);
            mass_upd_impl(ql, qr, self->get_root(), 0, self->n, forward<Args>(args)...);
        }
    };
};

struct BaseSegTree {
    [[gnu::always_inline]] void init() {}
    template<class... Args>
    [[gnu::always_inline]] void upd(Args&&...) {}
    [[no_unique_address]] DoNothingFunc join;
};

#define SEG_TREE_HELPERS(SegTree) \
    [[gnu::always_inline]] void push(size_t v, size_t l, size_t r) { \
        (Policies<SegTree<T, Policies...>>::push(v, l, r), ...); \
    } \
    template<class... Args> \
    [[gnu::always_inline]] void upd(size_t p, Args&&... args) { \
        constexpr int max_prior = max({Policies<SegTree<T, Policies...>>::update_priority...}); \
        auto select = []<class X>(X x) { \
            if constexpr (X::update_priority >= max_prior) { \
                return x; \
            } else { \
                return BaseSegTree(); \
            } \
        }; \
        (decltype(select(declval<Policies<SegTree<T, Policies...>>>()))::upd(p, forward<Args>(args)...), ...); \
    } \
    template<class P> \
    typename P::template type<SegTree<T, Policies...>> &as(const P&) { \
        return static_cast<typename P::template type<SegTree<T, Policies...>> &>(*this); \
    } \
    template<class... Args> \
    [[gnu::always_inline]] void join(T &res, const T &a, const T &b) { \
        constexpr auto max_prior = max({Policies<SegTree<T, Policies...>>::update_priority...}); \
        auto select = []<class X>(X x) { \
            if constexpr (X::update_priority >= max_prior) { \
                return x; \
            } else { \
                return BaseSegTree(); \
            } \
        }; \
        (assign_or_call( \
            static_cast<decltype(select(declval<Policies<SegTree<T, Policies...>>>())) *>(this)->join, \
            res, a, b), ...); \
    } \
    [[gnu::always_inline]] auto init(size_t l, size_t r) { \
        constexpr int max_prior = max({Policies<SegTree<T, Policies...>>::init_priority...}); \
        auto select = []<class X, class... Args>(auto select, X x, Args... args) { \
            if constexpr (X::init_priority >= max_prior) { \
                return x; \
            } else { \
                return select(select, forward<Args>(args)...); \
            } \
        }; \
        return decltype(select(select, declval<Policies<SegTree<T, Policies...>>>()...))::init(l, r); \
    } \

template<class T, template<class> class... Policies>
struct SegTree: BaseSegTree, public Policies<SegTree<T, Policies...>>... {
    size_t n;
    vector<T> a;
    SegTree(size_t n, Policies<SegTree<T, Policies...>>... policies)
        : Policies<SegTree<T, Policies...>>(policies)...
        , n(n)
    {
        size_t sz = 1;
        while (sz < n)
            sz <<= 1;
        a.resize(sz * 2 - 1);
        constexpr int max_prior = max({Policies<SegTree<T, Policies...>>::init_priority...});
        auto select = []<class X>(X x) {
            if constexpr (X::init_priority >= max_prior) {
                return x;
            } else {
                return BaseSegTree();
            }
        };
        (decltype(select(policies))::init(), ...);
    }
    [[gnu::always_inline, gnu::const]] T &get_val(T &x) {
        return x;
    }
    [[gnu::always_inline, gnu::const]] size_t get_left(size_t v, size_t, size_t) const {
        return v * 2 + 1;
    }
    [[gnu::always_inline, gnu::const]] size_t get_right(size_t v, size_t, size_t) const {
        return v * 2 + 2;
    }
    [[gnu::always_inline, gnu::const]] size_t get_root() const {
        return 0;
    }
    SEG_TREE_HELPERS(SegTree)
};

template<class T, class... Policies>
SegTree<T, Policies::template type...> make_segtree(size_t n, Policies... policies) {
    return SegTree<T, Policies::template type...>(
        n,
        typename Policies::template type<SegTree<T, Policies::template type...>>(policies)...);
}

template<class T, template<class> class... Policies>
struct LazySegTree: BaseSegTree, public Policies<LazySegTree<T, Policies...>>... {
    struct Node {
        constexpr static unsigned null = numeric_limits<unsigned>::max();
        unsigned left = null, right = null;
        T val;
        Node(T &&val): val(val) {}
    };
    size_t n;
    vector<Node> a;
    LazySegTree(size_t n, Policies<LazySegTree<T, Policies...>>... policies)
        : Policies<LazySegTree<T, Policies...>>(policies)...
        , n(n)
    {}
    [[gnu::always_inline, gnu::const]] T &get_val(Node &node) {
        return node.val;
    }
    [[gnu::always_inline]] size_t get_left(size_t v, size_t l, size_t r) {
        if (a[v].left == Node::null) {
            a[v].left = a.size();
            a.eb(init(l, r));
        }
        return a[v].left;
    }
    [[gnu::always_inline]] size_t get_right(size_t v, size_t l, size_t r) {
        if (a[v].right == Node::null) {
            a[v].right = a.size();
            a.eb(init(l, r));
        }
        return a[v].right;
    }
    [[gnu::always_inline]] size_t get_root() {
        if (a.empty())
            a.eb(init(0, n));
        return 0;
    }
    SEG_TREE_HELPERS(LazySegTree)
};

template<class T, class... Policies>
LazySegTree<T, Policies::template type...> make_lazy_segtree(size_t n, Policies... policies) {
    return LazySegTree<T, Policies::template type...>(
        n,
        typename Policies::template type<LazySegTree<T, Policies::template type...>>(policies)...);
}

#undef SEG_TREE_HELPERS

#ifdef LOCAL
#pragma GCC diagnostic pop
#endif

}; // namespace segtree

#if !defined NOGNU && __x86_64__

namespace std {

template<>
struct make_unsigned<__int128> {
    using type = unsigned __int128;
};

template<>
struct make_unsigned<unsigned __int128> {
    using type = unsigned __int128;
};

}; // namespace std

#endif

template<class T, auto c>
constexpr T const_in_type = c;

template<class T_ = int, auto &mod = const_in_type<make_unsigned_t<T_>, M>, class U_ = ll>
struct ModInt {
    using T = make_unsigned_t<T_>;
    using U = make_unsigned_t<U_>;
    static_assert(sizeof(U) > sizeof(T));
    T x{};
    [[gnu::always_inline]] constexpr ModInt() {}
    [[gnu::always_inline]] constexpr ModInt(T x): x(x) {}
    template<
        class T2_,
        class U2_,
        class = enable_if_t<sizeof(T) == sizeof(T2_), void>
    >
    [[gnu::always_inline]] constexpr ModInt(ModInt<T2_, mod, U2_> other): x(other.x) {}
    [[nodiscard, gnu::always_inline, gnu::pure]]
    [[gnu::always_inline]] constexpr auto operator<=>(const ModInt&) const = default;
    [[nodiscard, gnu::always_inline, gnu::pure]] constexpr ModInt operator+(ModInt other) const {
        auto selector = []() {
            if constexpr (__builtin_constant_p(mod))
                return conditional_t<U(mod) + U(mod) == U(mod + mod), T, U>{};
            else
                return U{};
        };
        using imm_type = decltype(selector());
        auto res = imm_type(x) + imm_type(other.x);
        res -= res >= mod ? mod : 0;
        return ModInt(res);
    }
    [[gnu::always_inline]] constexpr ModInt &operator+=(ModInt other) {
        return *this = *this + other;
    }
    [[nodiscard, gnu::always_inline, gnu::pure]] constexpr ModInt operator-() const {
        return ModInt(x ? mod - x : 0);
    }
    [[nodiscard, gnu::always_inline, gnu::pure]] constexpr ModInt operator-(ModInt other) const {
        return *this + -other;
    }
    [[gnu::always_inline]] constexpr ModInt &operator-=(ModInt other) {
        return *this = *this - other;
    }
    [[nodiscard, gnu::always_inline, gnu::pure]] constexpr ModInt operator*(ModInt other) const {
        auto selector = []() {
            if constexpr (__builtin_constant_p(mod))
                return conditional_t<U(mod) * U(mod) == U(mod * mod), T, U>{};
            else
                return U{};
        };
        using imm_type = decltype(selector());
        auto res = imm_type(x) * imm_type(other.x);
        return ModInt(res % mod);
    }
    [[gnu::always_inline]] constexpr ModInt &operator*=(ModInt other) {
        return *this = *this * other;
    }
    template<class Int>
    [[nodiscard, gnu::pure, gnu::always_inline]] constexpr ModInt pow(Int p) const {
        #ifdef NOGNU
        if constexpr (1)
        #else
        if (is_constant_evaluated())
        #endif
        {
            ModInt x = *this;
            ModInt ans = 1;
            while (p) {
                if (p & 1)
                    ans *= x;
                x *= x;
                p >>= 1;
            }
            return ans;
        } else {
            return __gnu_cxx::power(*this, p);
        }
    }
    template<auto p>
    [[nodiscard, gnu::pure, gnu::always_inline]] constexpr ModInt pow() const {
        static_assert(is_integral_v<decltype(p)> && p >= 0);
        if constexpr (p == 0) {
            return {1};
        } else if constexpr (p % 2) {
            return *this * pow<p - 1>();
        } else {
            auto t = pow<p / 2>();
            return t * t;
        }
    }
    [[nodiscard, gnu::always_inline, gnu::pure]] constexpr ModInt inv() const {
        if constexpr (__builtin_constant_p(mod))
            return pow<mod - 2>();
        else
            return pow(mod - 2);
    }
    [[nodiscard, gnu::always_inline, gnu::pure]] constexpr ModInt operator/(ModInt other) const {
        return *this * other.inv();
    }
    [[gnu::always_inline]] constexpr ModInt &operator/=(ModInt other) {
        return *this = *this / other;
    }
};

struct fictive {};

template<class F>
[[gnu::always_inline]] inline auto cond_cmp(F &&f) {
    return [f]<class A, class B>[[gnu::always_inline]](const A &a, const B &b)->bool {
        if constexpr (is_same_v<A, fictive>) {
            return 0;
        } else if constexpr (is_same_v<B, fictive>) {
            return !f(a);
        } else {
            return f(a) < f(b);
        }
    };
}

template<class F, class T1, class T2, size_t... i>
[[gnu::always_inline]] inline enable_if_t<tuple_size<T1>::value == tuple_size<T2>::value, void>
elementwise_apply(F f, T1 &x, const T2 &y, index_sequence<i...>) {
    (((void) f(get<i>(x), get<i>(y))), ...);
}

template<class F, class T1, class T2, size_t... i>
[[gnu::always_inline]] inline auto
elementwise_operaton(F f, T1 x, const T2 &y, index_sequence<i...>)
requires(tuple_size<T1>::value == tuple_size<T2>::value)
{
    if constexpr (tuple_size<T1>::value == 2)
        return pair(f(get<i>(x), get<i>(y))...);
    else
        return tuple(f(get<i>(x), get<i>(y))...);
}

template<class F, class T1, class T2, size_t... i>
[[gnu::always_inline]] inline enable_if_t<is_tuple_like_v<T1> && !is_tuple_like_v<T2>, void>
elementwise_apply(F f, T1 &x, const T2 &y, index_sequence<i...>) {
    (((void) f(get<i>(x), y)), ...);
}

template<class F, class T1, class T2, size_t... i>
[[gnu::always_inline]] inline auto
elementwise_operaton(F f, T1 x, const T2 &y, index_sequence<i...>)
requires(is_tuple_like_v<T1> && !is_tuple_like_v<T2>)
{
    if constexpr (tuple_size<T1>::value == 2)
        return pair(f(get<i>(x), y)...);
    else
        return tuple(f(get<i>(x), y)...);
}

namespace std {

template<tuple_like T1, class T2>
[[gnu::always_inline]] inline T1 &operator+=(T1 &&x, T2 &&y) {
    elementwise_apply(
        lambda(x += y, auto &x, auto &y),
        forward<T1>(x), forward<T2>(y),
        make_index_sequence<tuple_size_v<remove_reference_t<T1>>>{}
    );
    return x;
}

template<tuple_like T1, class T2>
[[gnu::always_inline, nodiscard]] inline auto operator+(T1 &&x, T2 &&y) {
    return elementwise_operaton(
        lambda(x + y, auto &&x, auto &&y),
        std::forward<T1>(x), std::forward<T2>(y),
        make_index_sequence<tuple_size_v<remove_reference_t<T1>>>{}
    );
}

template<class T1, class T2>
[[gnu::always_inline, nodiscard]] inline enable_if_t<is_tuple_like_v<T1> && !is_tuple_like_v<T2>, T1>
operator+(const T2 &y, T1 x) {
    x += y;
    return x;
}

template<tuple_like T1, class T2>
[[gnu::always_inline]] inline T1 &operator-=(T1 &&x, T2 &&y) {
    elementwise_apply(
        lambda(x -= y, auto &x, auto &y),
        forward<T1>(x), forward<T2>(y),
        make_index_sequence<tuple_size_v<remove_reference_t<T1>>>{}
    );
    return x;
}

template<tuple_like T1, class T2>
[[gnu::always_inline, nodiscard]] inline auto operator-(T1 &&x, T2 &&y) {
    return elementwise_operaton(
        lambda(x - y, auto &&x, auto &&y),
        std::forward<T1>(x), std::forward<T2>(y),
        make_index_sequence<tuple_size_v<remove_reference_t<T1>>>{}
    );
}

template<class T1, class T2>
[[gnu::always_inline, nodiscard]] inline enable_if_t<is_tuple_like_v<T1> && !is_tuple_like_v<T2>, T1>
operator-(const T2 &y, T1 x) {
    x -= y;
    return x;
}

template<tuple_like T1, class T2>
[[gnu::always_inline]] inline T1 &operator*=(T1 &&x, T2 &&y) {
    elementwise_apply(
        lambda(x *= y, auto &x, auto &y),
        forward<T1>(x), forward<T2>(y),
        make_index_sequence<tuple_size_v<remove_reference_t<T1>>>{}
    );
    return x;
}

template<tuple_like T1, class T2>
[[gnu::always_inline, nodiscard]] inline auto operator*(T1 &&x, T2 &&y) {
    return elementwise_operaton(
        lambda(x * y, auto &&x, auto &&y),
        std::forward<T1>(x), std::forward<T2>(y),
        make_index_sequence<tuple_size_v<remove_reference_t<T1>>>{}
    );
}

template<class T1, class T2>
[[gnu::always_inline, nodiscard]] inline enable_if_t<is_tuple_like_v<T1> && !is_tuple_like_v<T2>, T1>
operator*(const T2 &y, T1 x) {
    x *= y;
    return x;
}

template<tuple_like T1, class T2>
[[gnu::always_inline]] inline T1 &operator/=(T1 &&x, T2 &&y) {
    elementwise_apply(
        lambda(x /= y, auto &x, auto &y),
        forward<T1>(x), forward<T2>(y),
        make_index_sequence<tuple_size_v<remove_reference_t<T1>>>{}
    );
    return x;
}

template<tuple_like T1, class T2>
[[gnu::always_inline, nodiscard]] inline auto operator/(T1 &&x, T2 &&y) {
    return elementwise_operaton(
        lambda(x / y, auto &&x, auto &&y),
        std::forward<T1>(x), std::forward<T2>(y),
        make_index_sequence<tuple_size_v<remove_reference_t<T1>>>{}
    );
}

template<class T1, class T2>
[[gnu::always_inline, nodiscard]] inline enable_if_t<is_tuple_like_v<T1> && !is_tuple_like_v<T2>, T1>
operator/(const T2 &y, T1 x) {
    x /= y;
    return x;
}

template<class T, auto &mod, class U>
ostream &operator<<(ostream &out, ModInt<T, mod, U> v) {
    return out << v.x;
}

template<class T, auto &mod, class U>
istream &operator>>(istream &in, ModInt<T, mod, U> &v) {
    return in >> v.x;
}

};

namespace geometry {

template<class T1, class T2, class T3>
[[nodiscard, gnu::always_inline, gnu::const]] inline bool near(T1 x, T2 y, T3 eps) {
    return abs(x - y) <= eps;
}

template <class T>
struct Point {
    T x, y;
    template <class U, std::enable_if_t<!std::is_same_v<U, std::common_type_t<T, U>>, bool> = true>
    [[gnu::always_inline]] explicit operator Point<U>() const noexcept {
        return Point<U>{static_cast<U>(x), static_cast<U>(y)};
    }
    template <class U, std::enable_if_t<std::is_same_v<U, std::common_type_t<T, U>>, bool> = true>
    [[gnu::always_inline]] operator Point<U>() const noexcept {
        return Point<U>{static_cast<U>(x), static_cast<U>(y)};
    }
    [[gnu::always_inline]] Point() = default;
    [[gnu::always_inline]] Point(T x, T y) : x(x), y(y) {}
    [[gnu::always_inline]] explicit Point(double radians): x(std::cos(radians)), y(std::sin(radians)) {}
    template <class V>
    [[gnu::always_inline]] explicit Point(const V& v) noexcept : x(v.x), y(v.y) {}
    [[gnu::always_inline]] Point& operator+=(const Point& other) {
        x += other.x;
        y += other.y;
        return *this;
    }
    [[gnu::always_inline]] Point& operator-=(const Point& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }
    template <class U>
    [[gnu::always_inline, nodiscard, gnu::pure]] Point<std::common_type_t<T, U>> operator+(
        const Point<U>& other) const noexcept {
        return {x + other.x, y + other.y};
    }
    template <class U>
    [[gnu::always_inline, nodiscard, gnu::pure]] Point<std::common_type_t<T, U>> operator-(
        const Point<U>& other) const noexcept {
        return {x - other.x, y - other.y};
    }
    [[gnu::always_inline, nodiscard, gnu::pure]] T lenSq() const noexcept { return x * x + y * y; }
    [[gnu::always_inline, nodiscard, gnu::pure]] auto len() const noexcept {
        return std::sqrt(lenSq());
    }
    [[gnu::always_inline, nodiscard, gnu::pure]] Point operator-() const noexcept {
        return Point{-x, -y};
    }
    [[gnu::always_inline]] Point& operator*=(T c) {
        x *= c;
        y *= c;
        return *this;
    }
    template <class Dummy = T, std::enable_if_t<std::is_floating_point_v<Dummy>, bool> = true>
    [[gnu::always_inline]] Point& operator/=(T c) {
        x /= c;
        y /= c;
        return *this;
    }
    template <class U>
    [[gnu::always_inline, nodiscard, gnu::pure]] auto rotate(U radians) const {
        using Ret = std::conditional_t<std::is_floating_point_v<T>, T, double>;
        using Ang = std::common_type_t<Ret, U>;
        auto sn = std::sin(static_cast<Ang>(radians));
        auto cs = std::cos(static_cast<Ang>(radians));
        return Point<Ret>{x * cs - y * sn, x * sn + y * cs};
    }
    [[gnu::always_inline, nodiscard, gnu::pure]] Point left() const noexcept {
        Point ans = *this;
        std::swap(ans.x, ans.y);
        ans.x = -ans.x;
        return ans;
    }
    [[gnu::always_inline, nodiscard, gnu::pure]] Point right() const noexcept {
        Point ans = *this;
        std::swap(ans.x, ans.y);
        ans.y = -ans.y;
        return ans;
    }
    [[gnu::always_inline, nodiscard, gnu::pure]] auto radians() const { return std::atan2(y, x); }
    [[gnu::always_inline, nodiscard, gnu::pure]] bool operator==(const Point& other) const noexcept {
        return x == other.x && y == other.y;
    }
};

template <class T1, class T2>
[[gnu::always_inline, nodiscard, gnu::pure]] inline Point<std::common_type_t<T1, T2>> operator*(
    const Point<T1>& v, T2 c) noexcept {
    return {v.x * c, v.y * c};
}

template <class T1, class T2>
[[gnu::always_inline, nodiscard, gnu::pure]] inline auto operator*(T2 c,
                                                                   const Point<T1>& v) noexcept {
    return v * c;
}

template <class T1, class T2>
[[gnu::always_inline, nodiscard, gnu::pure]] inline auto operator/(const Point<T1>& v,
                                                                   T2 c) noexcept {
    using Ret = std::conditional_t<std::is_integral_v<T1> && std::is_integral_v<T2>,
                                   double,
                                   std::common_type_t<T1, T2>>;
    return Point<Ret>{static_cast<Ret>(v.x) / c, static_cast<Ret>(v.y) / c};
}

template <class T1, class T2, class T3>
[[gnu::always_inline, nodiscard, gnu::pure]] inline bool near(const Point<T1>& a, const Point<T2>& b,
                                                              T3 eps) noexcept {
    return near(a.x, b.x, eps) && near(a.y, b.y, eps);
}

template <class T1, class T2>
[[gnu::always_inline, nodiscard, gnu::pure]] inline auto operator*(const Point<T1>& a,
                                                             const Point<T2>& b) noexcept {
    return a.x * b.x + a.y * b.y;
}

template <class T1, class T2>
[[gnu::always_inline, nodiscard, gnu::pure]] inline auto operator^(const Point<T1>& a,
                                                               const Point<T2>& b) noexcept {
    return a.x * b.y - a.y * b.x;
}

template <class T1, class T2>
[[gnu::always_inline, nodiscard, gnu::pure]] inline auto distSq(const Point<T1> &a,
                                                                const Point<T2> &b) noexcept {
    return (a - b).lenSq();
}

template <std::floating_point Ret = double, class T1 = void, class T2 = void>
[[gnu::always_inline, nodiscard, gnu::pure]] inline auto dist(const T1 &a, const T2 &b) {
    return std::sqrt((Ret) distSq(a, b));
}

template <class T>
[[gnu::always_inline]] inline std::ostream& operator<<(std::ostream& out, const Point<T>& v) {
    return out << v.x << " " << v.y;
}

template<class T>
[[gnu::always_inline]] inline istream &operator>>(istream &in, Point<T> &p) {
    return in >> p.x >> p.y;
}

template <class T>
struct Line {
    union {
        struct {
            T a, b;
        };
        Point<T> normal;
    };
    T c;

    [[gnu::always_inline]] Line(T a = 1, T b = 0, T c = 0) : a(a), b(b), c(c) {}
    [[gnu::always_inline]] Line(const Point<T> &norm, T c) : normal(norm), c(c) {}

    template<class U>
    Line(const Line<U> &other): normal(other.normal), c(other.c) {}

    [[gnu::always_inline, nodiscard, gnu::pure]] static Line fromTwoPoints(
        const Point<T>& p1, const Point<T>& p2) noexcept {
        Point<T> norm{p2.y - p1.y, p1.x - p2.x};
        return Line(norm, -(p1 * norm));
    }

    [[gnu::always_inline, nodiscard, gnu::pure]] static Line fromPointAndNormal(
        const Point<T>& p1, const Point<T>& norm) noexcept {
        return Line(norm, -(p1 * norm));
    }

    [[gnu::always_inline, nodiscard, gnu::pure]] static Line fromPointAndCollinear(
        const Point<T>& p1, const Point<T>& coll) noexcept {
        Point<T> norm{coll.y, -coll.x};
        return Line(norm, -(p1 * norm));
    }
    [[gnu::always_inline, nodiscard, gnu::pure]] bool operator==(const Line& other) const noexcept {
        return normal * other.c == other.normal * c;
    }
};

template <class T1, class T2, class T3>
[[gnu::always_inline, nodiscard, gnu::pure]] inline bool near(const Line<T1>& a, const Line<T2>& b,
                                                              T3 eps) noexcept {
    return near(a.normal * b.c, b.normal * a.c, eps);
}

template<floating_point Ret = double, class T = void>
[[gnu::always_inline, nodiscard, gnu::pure]] Ret dist(const Line<T> &l, const Point<T> &p) {
    return abs((Ret) (l.normal * p + l.c) / sqrt((Ret) l.normal.sqrlen()));
}

template<class T1, class T2>
[[gnu::always_inline, nodiscard, gnu::pure]] inline std::common_type_t<T1, T2> distDenomalized(const Line<T1> &l, const Point<T2> &p) {
    return (l.normal * p + l.c);
}

template<class T>
[[gnu::always_inline]] inline ostream &operator<<(ostream &out, const Line<T> &l) {
    return out << l.a << " " << l.b << " " << l.c;
}

template<class T>
[[gnu::always_inline]] inline istream &operator>>(istream &in, Line<T> &l) {
    return in >> l.a >> l.b >> l.c;
}

template<std::floating_point Ret = double, class T1 = void, class T2 = void, class T3 = int>
[[gnu::always_inline, nodiscard, gnu::pure]]
inline optional<vector<Point<Ret>>> intersect(const Line<T1> &l1, const Line<T2> &l2, T3 eps = 0) noexcept {
    if (near(l1.normal ^ l2.normal, 0, eps)) {
        if (l1 == l2) {
            return nullopt;
        }
        return vector<Point<Ret>>{};
    }
    if (near(l2.a, 0, eps)) {
        return intersect<Ret>(l2, l1, eps);
    }
    if (near(l1.a, 0, eps)) {
        Ret y = (Ret) -l1.c / l1.b;
        Ret x = (Ret) (-l2.b * y - l2.c) / l2.a;
        return vector{Point(x, y)};
    } else {
        Ret nb = l2.b - (Ret) l1.b * l2.a / l1.a;
        Ret nc = l2.c - (Ret) l1.c * l2.a / l1.a;
        Ret y = -nc / nb;
        Ret x = (Ret) (-l2.b * y - l2.c) / l2.a;
        return vector{Point(x, y)};
    }
}

template<class T>
struct Circle {
    Point<T> c;
    T r;
    [[gnu::always_inline, nodiscard, gnu::pure]] bool operator==(const Circle &other) const {
        return c == other.c && r == other.r;
    }
};

template<std::floating_point Ret = double, class T1 = void, class T2 = void, class T3 = int>
[[gnu::always_inline, nodiscard, gnu::pure]]
inline optional<vector<Point<Ret>>> intersect(const Circle<T1> &c, const Line<T2> &l, T3 eps = 0) noexcept {
    Ret d = dist<Ret>(l, c.c);
    if (d > c.r) {
        return vector<Point<Ret>>{};
    } else {
        Point<Ret> dn = l.normal;
        if (dist_denomalized(l, c.c) > 0)
            dn = -dn;
        dn /= sqrt((Ret) dn.sqrlen());
        dn *= d;
        Point<Ret> p = c.c;
        p += dn;
        dn = l.normal.left();
        dn /= sqrt(dn.sqrlen());
        dn *= sqrt(c.r * c.r - d * d);
        if (near(p + dn, p - dn, eps)) {
            return vector{p + dn};
        } else {
            return vector{p + dn, p - dn};
        }
    }
}

template<std::floating_point Ret = double, class T1 = void, class T2 = void, class T3 = int>
[[gnu::always_inline, nodiscard, gnu::pure]]
inline optional<vector<Point<Ret>>> intersect(Circle<T1> c1, const Circle<T2> &c2, T3 eps = 0) noexcept {
    if (c1 == c2) {
        return nullopt;
    }
    c2.c -= c1.c;
    auto ans = intersect<Ret>(c2, Line<T2>(-2 * c2.c.x, -2 * c2.c.y,
        c2.c.x * c2.c.x + c2.c.y * c2.c.y + c1.r * c1.r - c2.r * c2.r));
    for (auto &p : ans)
        p += c1.c;
    return ans;
}

template<class T>
[[gnu::always_inline]] inline ostream &operator<<(ostream &out, const Circle<T> &c) {
    return out << c.c << " " << c.r;
}

template<class T>
[[gnu::always_inline]] inline istream &operator>>(istream &in, Circle<T> &c) {
    return in >> c.c >> c.r;
}

template<class T, class T2 = int>
[[gnu::always_inline, nodiscard]] inline bool is_inside(const Point<T> &p, const vector<Point<T>> &polygon, T2 eps = 0) {
    double sum = 0;
    for (int i = 0; i < (int) polygon.size(); ++i) {
        auto a = polygon[i] - p;
        auto b = polygon[(i + 1 == (int) polygon.size()) ? 0 : i + 1] - p;
        if (a * b <= 0 && near(a ^ b, 0, eps)) {
            return 1;
        }
        sum += atan2(a ^ b, a * b);
    }
    return abs(sum) > 1;
}

template<class T>
[[gnu::always_inline, nodiscard]] inline T sqr2(const vector<Point<T>> &polygon) {
    T sum = 0;
    for (int i = 0; i < (int) polygon.size(); ++i) {
        sum += polygon[i] ^ polygon[(i + 1 == (int) polygon.size()) ? 0 : i + 1];
    }
    if (sum < 0)
        sum = -sum;
    return sum;
}

}; // geometry

template<class T, class V>
[[gnu::always_inline, nodiscard, gnu::const]] inline auto &as_array(V &v) {
    using Arr = array<T, sizeof(V) / sizeof(T)>;
    return *reinterpret_cast<conditional_t<is_const_v<V>, const Arr, Arr> *>(&v);
}

template<class V, bool allgined = false, class T = int>
[[gnu::always_inline, nodiscard]] inline V load_vec(T *ptr) {
    if constexpr (sizeof(V) == 32) {
        auto vptr = reinterpret_cast<__m256i *>(ptr);
        if constexpr (allgined) {
            return reinterpret_cast<V>(_mm256_load_si256(vptr));
        } else {
            return reinterpret_cast<V>(_mm256_lddqu_si256(vptr));
        }
    } else if constexpr (sizeof(V) == 16) {
        auto vptr = reinterpret_cast<__m128i *>(ptr);
        if constexpr (allgined) {
            return reinterpret_cast<V>(_mm_load_si128(vptr));
        } else {
            return reinterpret_cast<V>(_mm_loadu_si128(vptr));
        }
    } else {
        static_assert(is_void_v<V>);
    }
}

template<bool flush = false>
void print() {
    if constexpr (flush)
        cout << endl;
    else
        cout << '\n';
}

template<bool flush = false, class T>
void print(const T &x) {
    cout << x;
    print<flush>();
}

template<bool flush = false, class T, class... Args>
void print(const T &x, const Args & ...args) {
    cout << x << " ";
    print<flush>(args...);
}

template<class T>
void prints(const T &x) {
    cout << x << " ";
}

template<class T, class... Args>
void prints(const T &x, const Args & ...args) {
    cout << x << " ";
    prints(args...);
}

#define named(x) #x, "=", x

// #define NODEBUG
#if defined NODEBUG || !defined LOCAL
#define debug(...)
#define debugs(...)
#else
#define debug print<true>
#define debugs prints
#endif

// ---SOLUTION--- //

// using namespace segtree;
// using namespace geometry;

void run() {
    
}
