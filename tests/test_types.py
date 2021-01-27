from src.ir import types as tp, kotlin_types as kt


def test_parameterized_supertypes_simple():
    foo_tparam = tp.TypeParameter("T")
    foo_con = tp.TypeConstructor("Foo", [foo_tparam], [])

    type_param = tp.TypeParameter("K")

    bar_con = tp.TypeConstructor("Bar", [type_param],
                                 [foo_con.new([type_param])])
    bar = bar_con.new([kt.String])

    supertypes = bar.supertypes
    assert len(supertypes) == 1
    assert isinstance(supertypes[0], tp.ParameterizedType)
    assert supertypes[0].type_args == [kt.String]

    # Type constructor hasn't changed.
    assert bar_con.supertypes[0] == tp.ParameterizedType(
        foo_con, [type_param])


def test_parameterized_mix_type_arguments():
    foo_con = tp.TypeConstructor(
        "Foo", [tp.TypeParameter("T1"), tp.TypeParameter("T2")], [])

    type_param = tp.TypeParameter("T1")
    foo_parent = foo_con.new([kt.String, type_param])
    bar_con = tp.TypeConstructor("Bar", [type_param], [foo_parent])
    bar = bar_con.new([kt.Integer])

    supertypes = bar.supertypes
    assert len(supertypes) == 1
    assert supertypes[0].type_args == [kt.String, kt.Integer]
    assert bar_con.supertypes[0] == foo_parent


def test_parameterized_nested_params():
    foo_con = tp.TypeConstructor(
        "Foo", [tp.TypeParameter("T1")], [])
    bar_con = tp.TypeConstructor(
        "Bar", [tp.TypeParameter("T2")], [])
    type_param = tp.TypeParameter("T")
    bar_parent = bar_con.new([foo_con.new([type_param])])

    baz_con = tp.TypeConstructor("Baz", [type_param], [bar_parent])
    baz = baz_con.new([kt.Boolean])

    supertypes = baz.supertypes
    assert supertypes[0] == tp.ParameterizedType(
        bar_con, [tp.ParameterizedType(foo_con, [kt.Boolean])])

def test_parameterized_with_chain_inheritance():
    foo_con = tp.TypeConstructor(
        "Foo", [tp.TypeParameter("T1")], [])
    bar_con = tp.TypeConstructor(
        "Bar", [tp.TypeParameter("T1")],
        [foo_con.new([tp.TypeParameter("T1")])]
    )
    baz_con = tp.TypeConstructor(
        "Bar", [tp.TypeParameter("T1")],
        [bar_con.new([tp.TypeParameter("T1")])])

    baz = baz_con.new([kt.String])

    supertypes = baz.supertypes

    assert supertypes[0].name == "Bar"
    assert supertypes[0].type_args == [kt.String]
    assert supertypes[0].supertypes[0].name == "Foo"
    assert supertypes[0].supertypes[0].type_args == [kt.String]
    assert len(baz.get_supertypes()) == 3


def test_parameterized_with_chain_inheritance_and_nested():
    type_param = tp.TypeParameter("T")
    type_param2 = tp.TypeParameter("K")

    x_con = tp.TypeConstructor(
        "X", [type_param], [])
    z_con = tp.TypeConstructor(
        "Z", [type_param, type_param2], [x_con.new([type_param])])
    y_con = tp.TypeConstructor(
        "Y", [type_param], [z_con.new([kt.String, type_param])])
    w_con = tp.TypeConstructor(
        "W", [type_param], [y_con.new([type_param])])
    k_con = tp.TypeConstructor(
        "K", [type_param], [])
    r_con = tp.TypeConstructor(
        "R", [type_param], [k_con.new([type_param])])
    test_con = tp.TypeConstructor(
        "Test", [type_param, type_param2],
        [w_con.new([r_con.new([type_param2])])])

    test_type = test_con.new([kt.String, kt.Boolean])

    st = test_type.supertypes[0]
    assert st.name == "W"
    assert len(st.type_args) == 1
    assert st.type_args[0].name == "R"
    assert st.type_args[0].type_args == [kt.Boolean]
    assert st.type_args[0].supertypes[0] == \
        tp.ParameterizedType(k_con, [kt.Boolean])

    st = st.supertypes[0]
    assert st.name == "Y"
    assert st.type_args[0].name == "R"
    assert st.type_args[0].type_args == [kt.Boolean]
    assert st.type_args[0].supertypes[0] == \
        tp.ParameterizedType(k_con, [kt.Boolean])

    st = st.supertypes[0]
    assert st.name == "Z"
    assert st.type_args[0] == kt.String
    assert st.type_args[1].name == "R"
    assert st.type_args[1].type_args == [kt.Boolean]
    assert st.type_args[1].supertypes[0] == \
        tp.ParameterizedType(k_con, [kt.Boolean])

    st = st.supertypes[0]
    assert st.name == "X"
    assert st.type_args == [kt.String]


def test_parameterized_with_bound_abstract():
    type_param = tp.TypeParameter("T")
    type_param2 = tp.TypeParameter("K", bound=type_param)

    x_con = tp.TypeConstructor("X", [type_param, type_param2], [])
    x = x_con.new([kt.Any, kt.String])

    assert x.supertypes == []
    assert x.t_constructor.type_parameters == \
        [type_param, tp.TypeParameter("K", bound=kt.Any)]
