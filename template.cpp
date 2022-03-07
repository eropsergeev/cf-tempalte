#ifdef LOCAL
    #define _GLIBCXX_DEBUG
    #define _GLIBCXX_DEBUG_PEDANTIC
#endif
//#define NOGNU
#ifndef LOCAL
    #pragma GCC optimize("Ofast")
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

using ld = long double;
using pii = std::pair<int, int>;
using pll = std::pair<ll, ll>;

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

const unsigned ll M1 = 4294967291, M2 = 4294967279, M = 998244353;
const ld EPS = 1e-8;

#ifndef M_PI
    const ld M_PI = acos(-1);
#endif // M_PI

using namespace std;

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
    [[nodiscard, gnu::pure]] constexpr operator T() const {
        if constexpr (is_floating_point_v<T>) {
            if constexpr (neg) {
                return -numeric_limits<T>::infinity;
            } else {
                return numeric_limits<T>::infinity;
            }
        } else if constexpr (neg) {
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
[[gnu::always_inline]] inline bool mini(T1 &a, T2 b) {
    if (a > b) {
        a = b;
        return 1;
    }
    return 0;
}

template<class T1, class T2>
[[gnu::always_inline]] inline bool maxi(T1 &a, T2 b) {
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
    _mm_setcsr(_mm_getcsr() & ~(0x0080 | 0x0200));
    #ifndef LOCAL
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

namespace std {

template<std::ranges::range T>
[[gnu::always_inline]] inline
enable_if_t<!is_same_v<T, string> && !is_array_v<T>
#ifndef NOGNU
    && !is_same_v<T, rope<char>>
#endif
, ostream> &
operator<<(ostream &out, const T &arr) {
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
[[gnu::always_inline]] inline 
enable_if_t<!is_same_v<T, string> && !is_array_v<T>
#ifndef NOGNU
    && !is_same_v<T, rope<char>>
#endif
, istream>&
operator>>(istream &in, T &arr) {
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

template<class T, size_t pos = 0>
[[gnu::always_inline]] inline 
enable_if_t<is_tuple_like_v<T> && !std::ranges::range<T>, istream>&
operator>>(istream &in, T &x) {
    read_tuple<T, 0>(in, x);
    return in;
}

template<class T>
[[gnu::always_inline]] inline 
enable_if_t<is_tuple_like_v<T> && !std::ranges::range<T>, ostream>&
operator<<(ostream &out, const T &x) {
    write_tuple<T, 0>(out, x);
    return out;
}

};

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

template<class T_ = int, make_unsigned_t<T_> mod = M, class U_ = ll>
struct ModInt {
    using T = make_unsigned_t<T_>;
    using U = make_unsigned_t<U_>;
    static_assert(sizeof(U) > sizeof(T));
    T x;
    constexpr ModInt() {}
    constexpr ModInt(T x): x(x) {}
    template<
        class T2_,
        class U2_,
        class = enable_if_t<sizeof(T) == sizeof(T2_), void>
    >
    constexpr ModInt(ModInt<T2_, mod, U2_> other): x(other.x) {}
    [[nodiscard, gnu::always_inline, gnu::pure]]
    constexpr auto operator<=>(const ModInt&) const = default;
    [[nodiscard, gnu::always_inline, gnu::pure]] constexpr ModInt operator+(ModInt other) const {
        using imm_type = conditional_t<U(mod) + U(mod) == U(mod + mod), T, U>;
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
    constexpr ModInt &operator-=(ModInt other) {
        return *this = *this - other;
    }
    [[nodiscard, gnu::always_inline, gnu::pure]] constexpr ModInt operator*(ModInt other) const {
        using imm_type = conditional_t<U(mod) * U(mod) == U(mod * mod), T, U>;
        auto res = imm_type(x) * imm_type(other.x);
        return ModInt(res % mod);
    }
    constexpr ModInt &operator*=(ModInt other) {
        return *this = *this * other;
    }
    template<class Int>
    [[nodiscard, gnu::pure]] constexpr ModInt pow(Int p) const {
        #ifdef NOGNU
        if constexpr (1)
        #else
        if (is_constant_evaluated())
        #endif
        {
            if (!p) {
                return 1;
            } else if (p & 1) {
                return *this * pow(p - 1);
            } else {
                auto t = pow(p >> 1);
                return t * t;
            }
        } else {
            return __gnu_cxx::power(*this, p);
        }
    }
    [[nodiscard, gnu::always_inline, gnu::pure]] constexpr ModInt inv() const {
        return pow(mod - 2);
    }
    [[nodiscard, gnu::always_inline, gnu::pure]] constexpr ModInt operator/(ModInt other) const {
        return *this * other.inv();
    }
    constexpr ModInt &operator/=(ModInt other) {
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

template<class T, make_unsigned_t<T> mod, class U>
ostream &operator<<(ostream &out, ModInt<T, mod, U> v) {
    return out << v.x;
}

template<class T, make_unsigned_t<T> mod, class U>
istream &operator>>(istream &in, ModInt<T, mod, U> &v) {
    return in >> v.x;
}

};

namespace geometry {

template<class T>
bool eq(T a, T b) {
    if constexpr (is_floating_point<T>()) {
        return fabs(a - b) < EPS;
    } else {
        return a == b;
    }
}

template<class T>
struct Point {
    T x, y;
    Point(T x = 0, T y = 0): x(x), y(y) {}
    template<class U>
    inline operator Point<U>() const {
        return Point<U>(x, y);
    }
    inline Point &operator+=(const Point &other) {
        x += other.x;
        y += other.y;
        return *this;
    }
    inline Point &operator-=(const Point &other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }
    [[gnu::always_inline, nodiscard, gnu::pure]] inline Point operator+(const Point &other) const {
        Point tmp = *this;
        tmp += other;
        return tmp;
    }
    [[gnu::always_inline, nodiscard, gnu::pure]] inline Point operator-(const Point &other) const {
        Point tmp = *this;
        tmp -= other;
        return tmp;
    }
    [[gnu::always_inline, nodiscard, gnu::pure]] inline T sqrlen() const {
        return x * x + y * y;
    }
    [[gnu::always_inline, nodiscard, gnu::pure]] inline T operator*(const Point &other) const {
        return x * other.x + y * other.y;
    }
    [[gnu::always_inline, nodiscard, gnu::pure]] inline T operator^(const Point &other) const {
        return x * other.y - y * other.x;
    }
    [[gnu::always_inline, nodiscard, gnu::pure]] inline Point operator-() const {
        return Point(-x, -y);
    }
    inline Point &operator*=(T c) {
        x *= c;
        y *= c;
        return *this;
    }
    inline Point &operator/=(T c) {
        x /= c;
        y /= c;
        return *this;
    }
    [[gnu::always_inline, nodiscard, gnu::pure]] inline Point operator*(T c) const {
        Point tmp = *this;
        tmp *= c;
        return tmp;
    }
    [[gnu::always_inline, nodiscard, gnu::pure]] inline Point operator/(T c) const {
        Point tmp = *this;
        tmp /= c;
        return tmp;
    }
    [[gnu::always_inline, nodiscard, gnu::pure]] bool operator==(const Point &other) const {
        return eq(x, other.x) && eq(y, other.y);
    }
};

template<class T>
[[gnu::always_inline, nodiscard, gnu::pure]] T sqrdist(const Point<T> &a, const Point<T> &b) {
    return (a - b).sqrlen();
}

template<class T1, class T2>
[[gnu::always_inline, nodiscard, gnu::pure]] double dist(const T1 &a, const T2 &b) {
    return sqrt(sqrdist(a, b));
}

template<class T>
ostream &operator<<(ostream &out, const Point<T> &p) {
    return out << p.x << " " << p.y;
}

template<class T>
istream &operator>>(istream &in, Point<T> &p) {
    return in >> p.x >> p.y;
}

template<class T>
struct Line {
    union {
        struct {
            T a, b;
        };
        Point<T> normal;
    };
    T c;
    Line(T a = 1, T b = 0, T c = 0): a(a), b(b), c(c) {}
    Line(const Point<T> &p1, const Point<T> &p2): a(p2.y - p1.y), b(p1.x - p2.x), c(-(p1 * normal)) {}
    [[gnu::always_inline, nodiscard, gnu::pure]] bool operator==(const Line &other) const {
        return eq(c * sqrt(other.normal.sqrlen()), other.c * sqrt(normal.sqrlen()));
    }
};

template<class T>
[[gnu::always_inline, nodiscard, gnu::pure]] double dist(const Line<T> &l, const Point<T> &p) {
    return fabs((double) (l.normal * p + l.c) / sqrt(l.normal.sqrlen()));
}

template<class T>
[[gnu::always_inline, nodiscard, gnu::pure]] T dist_denomalized(const Line<T> &l, const Point<T> &p) {
    return (l.normal * p + l.c);
}

template<class T>
ostream &operator<<(ostream &out, const Line<T> &l) {
    return out << l.a << " " << l.b << " " << l.c;
}

template<class T>
istream &operator>>(istream &in, Line<T> &l) {
    return in >> l.a >> l.b >> l.c;
}

template<class T>
optional<vector<Point<double>>> intersect(const Line<T> &l1, const Line<T> &l2) {
    if (eq(l1.normal ^ l2.normal, (T) 0)) {
        if (l1 == l2) {
            return nullopt;
        }
        return vector<Point<double>>{};
    }
    if (eq(l2.a, (T) 0)) {
        return intersect(l2, l1);
    }
    if (eq(l1.a, (T) 0)) {
        double y = (double) -l1.c / l1.b;
        double x = (double) (-l2.b * y - l2.c) / l2.a;
        return vector{Point(x, y)};
    } else {
        double nb = l2.b - (double) l1.b * l2.a / l1.a;
        double nc = l2.c - (double) l1.c * l2.a / l1.a;
        double y = -nc / nb;
        double x = (double) (-l2.b * y - l2.c) / l2.a;
        return vector{Point(x, y)};
    }
}

template<class T>
struct Circle {
    Point<T> c;
    T r;
    [[gnu::always_inline, nodiscard, gnu::pure]] bool operator==(const Circle &other) const {
        return c == other.c && eq(r, other.r);
    }
};

template<class T>
[[gnu::always_inline, nodiscard]]
optional<vector<Point<double>>> intersect(const Circle<T> &c, const Line<T> &l) {
    double d = dist(l, c.c);
    if (d > c.r) {
        return vector<Point<double>>{};
    } else {
        Point<double> dn = l.normal;
        if (dist_denomalized(l, c.c) > 0)
            dn = -dn;
        dn /= sqrt(dn.sqrlen());
        dn *= d;
        Point<double> p = c.c;
        p += dn;
        dn = l.normal;
        swap(dn.x, dn.y);
        dn.x = -dn.x;
        dn /= sqrt(dn.sqrlen());
        dn *= sqrt(c.r * c.r - d * d);
        if (p + dn == p - dn) {
            return vector{p + dn};
        } else {
            return vector{p + dn, p - dn};
        }
    }
}

template<class T>
[[gnu::always_inline, nodiscard]]
optional<vector<Point<double>>> intersect(Circle<T> c1, Circle<T> c2) {
    if (c1 == c2) {
        return nullopt;
    }
    c2.c -= c1.c;
    auto ans = intersect(c2, Line<T>(-2 * c2.c.x, -2 * c2.c.y,
        c2.c.x * c2.c.x + c2.c.y * c2.c.y + c1.r * c1.r - c2.r * c2.r));
    for (auto &p : ans)
        p += c1.c;
    return ans;
}

template<class T>
ostream &operator<<(ostream &out, const Circle<T> &c) {
    return out << c.c << " " << c.r;
}

template<class T>
istream &operator>>(istream &in, Circle<T> &c) {
    return in >> c.c >> c.r;
}

template<class T>
[[gnu::always_inline, nodiscard]] bool is_inside(const Point<T> &p, const vector<Point<T>> &polygon) {
    double sum = 0;
    for (int i = 0; i < (int) polygon.size(); ++i) {
        auto a = polygon[i] - p;
        auto b = polygon[(i + 1 == (int) polygon.size()) ? 0 : i + 1] - p;
        if (a * b <= 0 && eq(a ^ b, (T) 0)) {
            return 1;
        }
        sum += atan2(a ^ b, a * b);
    }
    return fabs(sum) > 1;
}

template<class T>
[[gnu::always_inline, nodiscard]] T sqr2(const vector<Point<T>> &polygon) {
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

template<class T>
void print(const T &x) {
    cout << x << endl;
}

template<class T, class... Args>
void print(const T &x, const Args & ...args) {
    cout << x << " ";
    print(args...);
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
#define debug print
#define debugs prints
#endif

// ---SOLUTION--- //

// using namespace segtree;
// using namespace geometry;

void run() {
    
}