from src.ir import ast, rust_types as rt, types as tp
from src.translators.base import BaseTranslator

#modify maybe
def append_to(visit):
    def inner(self, node):
        self._nodes_stack.append(node)
        res = visit(self, node)
        self._nodes_stack.pop()
        #??? handle main function here
        self._children_res.append(res)
    return inner

class RustTranslator(BaseTranslator):
    filename = 'main.rs'
    incorrect_filename = 'incorrect.rs'
    executable = 'main'

    def __init__(self, package = None, options = {}):
        super().__init__(package, options)
        self._children_res = []
        self.indent = 0
        self.context = None
        self.types = []
        self._generator = None
        self._cast_number = False
        self.is_unit = False
        self.is_lambda = False
        self._nodes_stack = []

    def _reset_state(self):
        self._children_res = []
        self.indent = 0
        self.context = None
        self.types = []
        self._generator = None
        self._cast_number = False
        self.is_unit = False
        self.is_lambda = False
        self._nodes_stack = []

    @staticmethod
    def get_filename():
        return RustTranslator.filename

    @staticmethod
    def get_incorrect_filename():
        return RustTranslator.incorrect_filename
        
    def type_arg2str(self, t_arg): #TODO
        pass

    def get_type_name(self, t): #TODO
        return t.get_name()

    def pop_children_res(self, children):
        len_c = len(children)
        if not len_c:
            return []
        res = self._children_res[-len_c:]
        self._children_res = self._children_res[:-len_c]
        return res

    def _parent_is_block(self):
        return isinstance(self._nodes_stack[-2], ast.Block)

    #??? might be wrong
    def _is_global_scope(self):
        return len(self._nodes_stack) == 1

    def visit_program(self, node):
        self.context = node.context
        children = node.children()
        for c in children:
            c.accept(self)
        if self.package:
            package_str = 'package ' + self.package + '\n'
        else:
            package_str = ''
        self.program = package_str + '\n\n'.join(
            self.pop_children_res(children))

    
    @append_to
    def visit_block(self, node):
        old_indent = self.indent
        children = node.children()
        is_unit = self.is_unit
        is_lambda = self.is_lambda
        self.is_unit = False
        self.is_lambda = False
        children_len = len(children)
        for i, c in enumerate(children):
            #casting if return statement is a number
            if i == children_len - 1:
                prev_cast_number = self._cast_number
                self._cast_number = True
                c.accept(self)
                self._cast_number = prev_cast_number
            else:
                c.accept(self)
        children_res = self.pop_children_res(children)
        res = "{\n"
        res += ";\n".join(children_res[:-1])
        if children_res[:-1]:
            res += ";\n"
        last_char = (
            ";"
            if is_unit
            else ""
        )
        if children_res:
            res +=  children_res[-1] + last_char + "\n" #+ \
                   #" " * self.indent
        else:
            res += " " * self.indent + last_char + "\n" + \
                   " " * self.indent
        res += " " * old_indent + "}"
        self.is_unit = is_unit
        self.is_lambda = is_lambda
        return res

    @append_to
    def visit_func_decl(self, node):
        old_indent = self.indent
        self.indent += 2
        children = node.children()
        prev_is_unit = self.is_unit
        self.is_unit = node.get_type() == rt.Unit #??? might be wrong/irrelevant
        prev_cast_number = self._cast_number
        is_expression = not isinstance(node.body, ast.Block)
        if is_expression:
            self._cast_number = True
        for c in children:
            c.accept(self)
        children_res = self.pop_children_res(children)
        param_res = [children_res[i] for i,_ in enumerate(node.params)]
        len_params = len(node.params)
        len_type_params = len(node.type_parameters)
        type_parameters_res = ", ".join(children_res[len_params:len_type_params + len_params])
        body_res = children_res[-1] if node.body else ""
        prefix = " " * old_indent
        type_params = "<" + type_parameters_res + ">" if type_parameters_res else ""
        res = prefix + "fn " + node.name + type_params + "(" + ", ".join(param_res) + ")"
        if node.ret_type:
            res += " -> " + self.get_type_name(node.ret_type)
        res += body_res #check if no other checks necessary
        self.indent = old_indent
        self.is_unit = prev_is_unit
        self._cast_number = prev_cast_number
        return res

    @append_to
    def visit_param_decl(self, node):
        param_type = node.param_type
        res = node.name + ": " + self.get_type_name(param_type) #??? handle vararg 
        return res

    @append_to
    def visit_func_call(self, node):
        old_indent = self.indent
        self.indent = 0
        children = node.children()
        for c in children:
            c.accept(self)
        self.indent = old_indent
        children_res = self.pop_children_res(children)
        type_args = (
            "::<" + ",".join([self.get_type_name(t) for t in node.type_args]) + ">"
            if not node.can_infer_type_args and node.type_args
            else ""
        )
        segs = node.func.rsplit(".", 1)
        if node.receiver:
            receiver_expr = children_res[0] + '.'
            func = node.func
            args = children_res[1:]
        else:
            receiver_expr, func = (
                ("", node.func)
                if len(segs) == 1
                else (segs[0], segs[1])
            )
            args = children_res
        res = "{indent}{receiver}{name}{type_args}({args})".format(
            indent=" " * self.indent,
            receiver=receiver_expr,
            name=func,
            type_args=type_args,
            args=", ".join(args)
        )
        return res

    @append_to
    def visit_call_argument(self, node):
        old_indent = self.indent
        self.indent = 0
        children = node.children()
        for c in children:
            c.accept(self)
        self.indent = old_indent
        children_res = self.pop_children_res(children)
        res = children_res[0]
        return res

    @append_to
    def visit_lambda(self, node):
        old_indent = self.indent
        is_expression = not isinstance(node.body, ast.Block)
        self.indent = 0 if is_expression else self.indent + 2
        children = node.children()
        prev_is_unit = self.is_unit
        prev_is_lambda = self.is_lambda
        self.is_lambda = True
        prev_cast_number = self._cast_number
        if is_expression:
            self._cast_number = True
        for c in children:
            c.accept(self)
        children_res = self.pop_children_res(children)
        self.indent = old_indent
        param_res = [children_res[i] for i,_ in enumerate(node.params)]
        body_res = children_res[-1] if node.body else ""
        '''ret_type_str = (
            ": " + self.get_type_name(node.ret_type)
            if node.ret_type
            else ""
        )'''
        res = "|{params}| {body}".format( #??? is ret even allowed in Rust
            params=", ".join(param_res),
            body=body_res,
            #ret = ret_type_str
        )
        self.indent = old_indent
        self.is_unit = prev_is_unit
        self.is_lambda = prev_is_lambda
        self._cast_number = prev_cast_number
        return res
    
    @append_to
    def visit_integer_constant(self, node):
        def get_cast_literal(integer_type, literal):
            if integer_type == rt.I8:
                return str(literal) + "as i8"
            if integer_type == rt.I16:
                return str(literal) + "as i16"
            if integer_type == rt.I32:
                return str(literal) + "as i32"
            if integer_type == rt.I64:
                return str(literal) + "as i64"
            if integer_type == rt.I128:
                return str(literal) + "as i128"
            if integer_type == rt.U8:
                return str(literal) + "as u8"
            if integer_type == rt.U16:
                return str(literal) + "as u16"
            if integer_type == rt.U32:
                return str(literal) + "as u32"
            if integer_type == rt.U64:
                return str(literal) + "as u64"
            if integer_type == rt.U128:
                return str(literal) + "as u128"
            #if integer_type == rt.Number:
            #    return str(literal) + "as i32" #??? is this default cast
            return str(literal)
        if not self._cast_number:
            return "{indent}{literal}{semicolon}".format(
                indent=" " * self.indent,
                literal=str(node.literal),
                semicolon=";" if self._parent_is_block() else ""
            )
        return "{indent}{cast_literal}{semicolon}".format(
            indent=" " * self.indent,
            cast_literal=get_cast_literal(node.integer_type, node.literal),
            semicolon="" if self._parent_is_block() else ""
        )

    @append_to
    def visit_real_constant(self, node):
        def get_cast_literal(real_type, literal):
            if real_type == rt.F32:
                return str(literal) + "as f32"
            if real_type == rt.F64:
                return str(literal) + "as f64"
            #if real_type == jt.Number:
            #    return "(Number) new Double(" + str(literal) + ")"
            return str(literal)

        if not self._cast_number:
            return "{indent}{literal}{semicolon}".format(
                indent=" " * self.indent,
                literal=str(node.literal),
                semicolon="" if self._parent_is_block() else ""
            )
        return "{indent}{cast_literal}{semicolon}".format(
            indent=" " * self.indent,
            cast_literal=get_cast_literal(node.real_type, node.literal),
            semicolon="" if self._parent_is_block() else ""
        )

    @append_to
    def visit_char_constant(self, node):
        return "{indent} '{literal}'{semicolon}".format(
            indent=" " * self.indent,
            literal=node.literal,
            semicolon="" if self._parent_is_block() else ""
        )

    @append_to
    def visit_string_constant(self, node):
        return "{indent}\"{literal}\"{semicolon}".format(
            indent=" " * self.indent,
            literal=node.literal,
            semicolon="" if self._parent_is_block() else ""
        )

    @append_to
    def visit_boolean_constant(self, node):
        return "{indent}{literal}{semicolon}".format(
            indent=" " * self.indent,
            literal=str(node.literal),
            semicolon="" if self._parent_is_block() else ""
        )
    
    @append_to
    def visit_binary_op(self, node):
        old_indent = self.indent
        self.indent = 0
        children = node.children()
        for c in children:
            c.accept(self)
        children_res = self.pop_children_res(children)
        res = "({left} {operator} {right}){semicolon}".format(
            #indent=" " * old_indent,
            left=children_res[0],
            operator=node.operator,
            right=children_res[1],
            semicolon=";" if self._parent_is_block() else ""
        )
        self.indent = old_indent
        return res

    def visit_logical_expr(self, node):
        self.visit_binary_op(node)

    def visit_equality_expr(self, node):
        self.visit_binary_op(node)

    def visit_comparison_expr(self, node):
        self.visit_binary_op(node)

    def visit_arith_expr(self, node):
        self.visit_binary_op(node)
    
    @append_to
    def visit_type_param(self, node):
        return node.name #??? handle covariance and contravariance
    
    @append_to
    def visit_variable(self, node):
        return "{indent}{name}{semicolon}".format(
            indent=" " * self.indent if self._parent_is_block() else "",
            name=node.name,
            semicolon="" if self._parent_is_block() else "" #???
        )

    @append_to
    def visit_var_decl(self, node):
        old_indent = self.indent
        prefix = " " * self.indent
        self.indent = 0
        children = node.children()
        prev = self._cast_number
        if node.var_type is None:
            self._cast_number = True
        for c in children:
            c.accept(self)
        children_res = self.pop_children_res(children)
        if self._is_global_scope():
            #static variables must have type annotation
            res = "static " + node.name + ": " + self.get_type_name(node.var_type) + " = " + children_res[0] + ";"
        else:
            mut = "let " if node.is_final else "let mut "
            res = prefix + mut + node.name
            if node.var_type is not None:
                res += ": " + self.get_type_name(node.var_type)
            res += " = " + children_res[0]
        self.indent = old_indent
        self._cast_number = prev
        return res
    
    @append_to
    def visit_conditional(self, node):
        old_indent = self.indent
        self.indent += 2
        children = node.children()
        for c in children:
            c.accept(self)
        children_res = self.pop_children_res(children)
        res = "{indent}(if {cond} {true} \n{indent}else {false})".format(
            indent=" " * old_indent,
            cond=children_res[0],#[self.indent:],
            true=children_res[1],
            false=children_res[2]
        )
        self.indent = old_indent
        return res
    
    #extend to handle field assignments
    @append_to
    def visit_assign(self, node):
        old_indent = self.indent
        prev_cast_number = self._cast_number
        self.indent = 0
        children = node.children()
        for c in children:
            c.accept(self)
        self.indent = old_indent
        children_res = self.pop_children_res(children)
        if node.receiver:
            receiver_expr = children_res[0] + '.' #??? consider BottomConstant
        else:
            receiver_expr = ''
        res = "{indent}{receiver_expr}{name} = {value}".format(
            indent=" " * old_indent,
            receiver_expr=receiver_expr,
            name=node.name,
            value=children_res[0]
        )
        self.indent = old_indent
        self._cast_number = prev_cast_number
        return res

    
