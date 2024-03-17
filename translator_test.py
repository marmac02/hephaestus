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
    test6()


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
    """ 
let a = 5;

let mut b = true;

let c = (if (a > 3)
{
    a
  }
else
{
  b = false;
    5
  });
    """
    program_str = utils.translate_program(translator, program)
    print(program_str)


def test6():
    translator = RustTranslator()
    context = Context()
    program = ast.Program(context, "scala")
    var_decl = ast.VariableDeclaration(name="a", inferred_type=Type("Integer"), expr=ast.IntegerConstant(literal=5, integer_type="Integer"))
    program.add_declaration(var_decl)
    bool_decl = ast.VariableDeclaration(name="b", inferred_type=Type("Boolean"), expr=ast.BooleanConstant(literal="true"), is_final=False)
    program.add_declaration(bool_decl)

    cond = ast.Variable("b")
    true_branch = ast.Block([ast.Conditional(ast.EqualityExpr(lexpr=ast.Variable(name="a"), rexpr=ast.IntegerConstant(literal=4, integer_type="Integer"), operator=ast.Operator("==")), \
        ast.Block([ast.IntegerConstant(5, "Integer")], False), ast.Block([ast.IntegerConstant(8, "Integer")], False), Type("Integer"))], False)
    #true_branch = ast.Block([ast.IntegerConstant(literal=5, integer_type="Integer")], False)
    false_branch = ast.Block([ast.Assignment(ast.Variable(name="b"), ast.BooleanConstant(literal="false")), ast.IntegerConstant(literal=5, integer_type="Integer")], False)
    if_stmt = ast.Conditional(cond, true_branch, false_branch, Type("Integer"))
    c_decl = ast.VariableDeclaration(name="c", inferred_type=Type("Integer"), expr=if_stmt)
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

if __name__ == "__main__":
    main()
