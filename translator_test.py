from src.ir import ast, types as tp, type_utils as tu
#from src.generators import Generator
from src.ir.context import Context
from src.ir.types import Type
from src.translators.rust import RustTranslator
from src.translators.scala import ScalaTranslator
from src import utils


def main():
    #test1()
    #test2()
    #test3()
    #test4()
    #test5()
    #test6()
    #test7()
    test8()


def test1():
    translator = RustTranslator()
    context = Context()
    program = ast.Program(context, "scala")
    var_decl = ast.VariableDeclaration(name="a", inferred_type=Type("Boolean"), expr=ast.BooleanConstant(literal="true"))
    program.add_declaration(var_decl)

    program_str = utils.translate_program(translator, program)
    print(program_str)

def test2():
    translator = RustTranslator()
    context = Context()
    program = ast.Program(context, "scala")
    var_decl = ast.VariableDeclaration(name="a", inferred_type=Type("Long"), expr=ast.IntegerConstant(literal=1, integer_type=Type("Long")))
    program.add_declaration(var_decl)

    program_str = utils.translate_program(translator, program)
    print(program_str)

def test3():
    translator = RustTranslator()
    context = Context()
    program = ast.Program(context, "scala")
    var_decl = ast.VariableDeclaration(name="a", inferred_type=Type("Double"), expr=ast.RealConstant(literal="1.0", real_type=Type("Double")))
    program.add_declaration(var_decl)

    program_str = utils.translate_program(translator, program)
    print(program_str)

def test4():
    translator = RustTranslator()
    context = Context()
    program = ast.Program(context, "scala")
    var_decl = ast.VariableDeclaration(name="a", inferred_type=Type("String"), expr=ast.StringConstant(literal="hello"))
    program.add_declaration(var_decl)

    program_str = utils.translate_program(translator, program)
    print(program_str)

def test5():
    translator = RustTranslator()
    context = Context()
    program = ast.Program(context, "scala")
    var_decl = ast.VariableDeclaration(name="a", inferred_type=Type("Integer"), expr=ast.IntegerConstant(literal=5, integer_type="Integer"))
    program.add_declaration(var_decl)
    bool_decl = ast.VariableDeclaration(name="b", inferred_type=Type("Boolean"), expr=ast.BooleanConstant(literal="true"), is_final=False)
    program.add_declaration(bool_decl)

    cond = ast.ComparisonExpr(lexpr=ast.Variable(name="a"), rexpr=ast.IntegerConstant(literal=3, integer_type="Integer"), operator=ast.Operator(">"))
    true_branch = ast.Block([ast.Variable("a")], False)
    false_branch = ast.Block([ast.Assignment(ast.Variable(name="b"), ast.BooleanConstant(literal="false")), ast.IntegerConstant(literal=5, integer_type="Integer")], False)
    if_stmt = ast.Conditional(cond, true_branch, false_branch, Type("Integer"))
    c_decl = ast.VariableDeclaration(name="c", inferred_type=Type("Integer"), expr=if_stmt)
    program.add_declaration(c_decl)

    program_str = utils.translate_program(translator, program)
    print(program_str)


def test6():
    translator = RustTranslator()
    context = Context()
    program = ast.Program(context, "scala")
    var_decl = ast.VariableDeclaration(name="a", inferred_type=Type("Integer"), expr=ast.IntegerConstant(literal=5, integer_type="Integer"), var_type=Type("i32"))
    program.add_declaration(var_decl)
    bool_decl = ast.VariableDeclaration(name="b", inferred_type=Type("Boolean"), expr=ast.BooleanConstant(literal="true"), is_final=False, var_type=Type("bool"))
    program.add_declaration(bool_decl)

    cond = ast.Variable("b")
    true_branch = ast.Block([ast.Conditional(ast.EqualityExpr(lexpr=ast.Variable(name="a"), rexpr=ast.IntegerConstant(literal=4, integer_type="Integer"), operator=ast.Operator("==")), \
        ast.Block([ast.IntegerConstant(5, "Integer")], False), ast.Block([ast.IntegerConstant(8, "Integer")], False), Type("Integer"))], False)
    #true_branch = ast.Block([ast.IntegerConstant(literal=5, integer_type="Integer")], False)
    false_branch = ast.Block([ast.Assignment(ast.Variable(name="b"), ast.BooleanConstant(literal="false")), ast.IntegerConstant(literal=5, integer_type="Integer")], False)
    if_stmt = ast.Conditional(cond, true_branch, false_branch, Type("Integer"))
    c_decl = ast.VariableDeclaration(name="c", inferred_type=Type("Integer"), expr=if_stmt, var_type=Type("i32"))
    program.add_declaration(c_decl)
    """
    let a = 5;

    let mut b = true;
    
    let c = (if b {
        (if (a == 4) {
            5
        }
      else {
            8
        })
      }
    else {
      b = false;
        5
      });
    """
    program_str = utils.translate_program(translator, program)
    print(program_str)

def test7():
    translator = RustTranslator()
    context = Context()
    program = ast.Program(context, "scala")


    cond = ast.Variable("b")
    #var_decl = ast.VariableDeclaration(name="var", inferred_type=Type("Integer"), expr=ast.IntegerConstant(literal=0, integer_type="Integer"), is_final=False)
    #program.add_declaration(var_decl)
    true_branch = ast.Block([ast.Conditional(ast.EqualityExpr(lexpr=ast.Variable(name="a"), rexpr=ast.IntegerConstant(literal=4, integer_type="Integer"), operator=ast.Operator("==")), \
        ast.Block([ast.IntegerConstant(5, "Integer")], False), ast.Block([ast.IntegerConstant(8, "Integer")], False), Type("Integer"))], False)
    false_branch = ast.Block([ast.Assignment(ast.Variable(name="b"), ast.BooleanConstant(literal="false")), ast.IntegerConstant(literal=5, integer_type="Integer")], False)
    if_stmt = ast.Conditional(cond, true_branch, false_branch, Type("Integer"))
    func_param1 = ast.ParameterDeclaration("a", Type("i32"))
    func_param2 = ast.ParameterDeclaration("word", Type("String"))
    func_body = ast.Block([ast.VariableDeclaration(name="b", inferred_type=Type("Boolean"), expr=ast.BooleanConstant("true"), is_final=False), \
        ast.VariableDeclaration(name="p", inferred_type=Type("Char"), expr=ast.CharConstant("p"), is_final=False), if_stmt], False)
    func = ast.FunctionDeclaration("someFun", [func_param1, func_param2], Type("i32"), func_body, 1)
    program.add_declaration(func)
    program_str = utils.translate_program(translator, program)
    print(program_str)

def test8():
    translator = RustTranslator()
    context = Context()
    program = ast.Program(context, "rust")
    cond = ast.Variable("b")
    field1 = ast.FieldDeclaration("field1", Type("i32"))
    struct = ast.StructDeclaration("SomeStruct", [field1])
    program.add_declaration(struct)

    func_param1 = ast.ParameterDeclaration("a", Type("i32"))
    func_param2 = ast.ParameterDeclaration("word", Type("String"))
    func = ast.FunctionDeclaration("someFun", [func_param1, func_param2], Type("i32"), None, 1)
    true_branch = ast.Block([ast.Conditional(ast.EqualityExpr(lexpr=ast.Variable(name="a"), rexpr=ast.IntegerConstant(literal=4, integer_type="Integer"), operator=ast.Operator("==")), \
        ast.Block([ast.IntegerConstant(5, "Integer")], False), ast.Block([ast.IntegerConstant(8, "Integer")], False), Type("Integer"))], False)
    false_branch = ast.Block([ast.Assignment(ast.Variable(name="b"), ast.BooleanConstant(literal="false")), ast.IntegerConstant(literal=5, integer_type="Integer")], False)
    if_stmt = ast.Conditional(cond, true_branch, false_branch, Type("Integer"))
    func2_body = ast.Block([ast.VariableDeclaration(name="b", inferred_type=Type("Boolean"), expr=ast.BooleanConstant("true"), is_final=False), \
        ast.VariableDeclaration(name="p", inferred_type=Type("Char"), expr=ast.CharConstant("p"), is_final=False), if_stmt], False)
    func2 = ast.FunctionDeclaration("someFun2", [func_param1], Type("i32"), func2_body, 1)
    trait = ast.TraitDeclaration("SomeTrait", [func], [func2])
    program.add_declaration(trait)

    program_str = utils.translate_program(translator, program)
    print(program_str)

if __name__ == "__main__":
    main()
