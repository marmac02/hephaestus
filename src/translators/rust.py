from src.ir import ast, rust_types as rt, types as tp
from src.translators.base import BaseTranslator
from src.ir.context import get_decl

#modify maybe
def append_to(visit):
    def inner(self, node):
        self._nodes_stack.append(node)
        res = visit(self, node)
        self._nodes_stack.pop()
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
        self.context = None

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
        self.context = None

    @staticmethod
    def get_filename():
        return RustTranslator.filename

    @staticmethod
    def get_incorrect_filename():
        return RustTranslator.incorrect_filename
        
    def type_arg2str(self, t_arg): #check if correct
        if not isinstance(t_arg, tp.WildCardType):
            return self.get_type_name(t_arg)
        return "type_arg2str ERROR"

    def get_type_name(self, t, is_bound=False): #check if correct
        if t.is_wildcard():
            t = t.get_bound_rec()
            return self.get_type_name(t)
        t_counstructor = getattr(t, 't_constructor', None)
        if not t_counstructor:
            return t.get_name()
        if t.is_function_type():
            func_type_name = t.t_constructor.get_name()
            func_type = func_type_name + "(" + ", ".join([self.get_type_name(t.type_args[ind]) for ind in range(len(t.type_args) - 1)]) + \
                 ") -> " + self.get_type_name(t.type_args[-1])
            if t.name[0].isupper() and not is_bound:
                return "Box<dyn " + func_type + ">"
            return func_type
        return "{}<{}>".format(t.name, ", ".join([self.get_type_name(t_arg) for t_arg in t.type_args]))

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

    def _is_func_in_trait(self, func_name):
        funcs = self.context.get_all_func_decl()
        #assert func_name in funcs.keys(), "Function not found in context"
        if not func_name in funcs.keys():
            #if function is not found then it is a variable of function type
            return False
        return funcs[func_name].trait_func

    def visit_program(self, node):
        self.context = node.context
        children = node.children()
        for c in children:
            c.accept(self)
        self.program = '\n\n'.join(self.pop_children_res(children))
        self._reset_state()

    
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
        #if self._is_func_in_trait(node.name):
        #    param_res = ["&self"] + param_res

        #type params implement Copy trait to avoid move issues CHANGE THIS
        #type_parameters_res = " :Copy, ".join(children_res[len_params:len_type_params + len_params])
        #type_parameters_res += " :Copy" if type_parameters_res else ""

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
        if isinstance(param_type, tp.SelfType): #parameter is self
            res = "&self"
        else:
            res = node.name + ": " + self.get_type_name(param_type) #??? handle vararg 
        return res

    @append_to
    def visit_func_ref(self, node):
        old_indent = self.indent
        self.indent = 0
        children = node.children()
        for c in children:
            c.accept(self)
        self.indent = old_indent
        children_res = self.pop_children_res(children)
        receiver = children_res[0] + "." if children_res else ""
        type_annotation = ""
        if node.signature.is_parameterized(): #annotation needed for parameterized functions
            type_annotation = " as " + self.get_type_name(node.signature)
        res = "{indent}{receiver}{name}{type_annotation}".format(
            indent=" " * self.indent,
            receiver=receiver, 
            name=node.func, 
            type_annotation=type_annotation)
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
            receiver_expr = "(" + children_res[0] + ")."
            func = node.func
            args = children_res[1:]
        else:
            receiver_expr, func = (
                ("", node.func)
                if len(segs) == 1
                else (segs[0] + '.', segs[1])
            )
            args = children_res
        #if receiver_expr == "" and self._is_func_in_trait(node.func):
        #    receiver_expr = "self."
        if self._is_func_in_trait(node.func):
            args = args[1:]
        if args is None:
            args = []
        res = "{indent}{left_bracket}{receiver}{name}{type_args}{right_bracket}({args})".format(
            indent=" " * self.indent,
            left_bracket="(" if node.is_ref_call and receiver_expr else "",
            receiver=receiver_expr,
            name=func,
            type_args=type_args,
            right_bracket=")" if node.is_ref_call and receiver_expr else "",
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
        res = "Box::new(move " + res + ")"
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

    #??? must be revisited
    @append_to
    def visit_bottom_constant(self, node):
        bottom = "unimplemented!()"
        return bottom

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

    #for now only String type supported
    @append_to
    def visit_string_constant(self, node):
        return "{indent}\"{literal}\".to_string(){semicolon}".format(
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
    def visit_array_expr(self, node):
        old_indent = self.indent
        self.indent = 0
        children = node.children()
        for c in children:
            c.accept(self)
        children_res = self.pop_children_res(children)
        self.indent = old_indent
        array_type = node.array_type
        type_annotation = " as Vec<{}>".format(self.get_type_name(array_type.type_args[0])) \
            if len(children_res) == 0 else ""
        res = "vec![" + ", ".join(children_res) + "]" + type_annotation
        return res
    
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
            semicolon="" if self._parent_is_block() else ""
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
        bound = ""
        if node.bound is not None:
            bound = " : " + self.get_type_name(node.bound, is_bound=True)
        return node.name + bound
    
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
            #static variables must have type annotation and be immutable (mutable only with unsafe)
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

    @append_to
    def visit_new(self, node):
        return "unimplemented!()"
    
    @append_to
    def visit_supertrait_instantiation(self, node):
        return self.get_type_name(node.trait_type)
    
    @append_to
    def visit_trait_decl(self, node):
        old_indent = self.indent
        self.indent += 2
        children = node.children()
        for c in children:
            c.accept(self)
        children_res = self.pop_children_res(children)
        function_signs_res = [children_res[i] for i, _ in enumerate(node.function_signatures)]
        len_function_signs = len(node.function_signatures)
        default_impls_res = [children_res[i + len_function_signs] 
                             for i, _ in enumerate(node.default_impls)]
        len_default_impls = len(node.default_impls)
        supertraits_res = [children_res[i + len_function_signs + len_default_impls] 
                           for i, _ in enumerate(node.supertraits)]
        len_supertraits = len(node.supertraits)
        type_parameters_res = [children_res[i + len_function_signs + len_default_impls + len_supertraits]
                               for i, _ in enumerate(node.type_parameters)]
        res = "{indent}trait {name}".format(
            indent=" " * old_indent,
            name=node.name
        )
        if type_parameters_res:
            res += "<" + ", ".join(type_parameters_res) + ">"
        if supertraits_res:
            res += " : " + " + ".join(supertraits_res)
        res += " {\n"
        if function_signs_res:
            res += ";\n".join(function_signs_res)
            res += ";\n"
        if default_impls_res:
            res += "\n".join(default_impls_res)
            res += "\n"
        res += " " * old_indent + "}"
        self.indent = old_indent
        return res

    @append_to
    def visit_struct_decl(self, node):
        old_indent = self.indent
        self.indent += 2
        children = node.children()
        for c in children:
            c.accept(self)
        children_res = self.pop_children_res(children)
        fields_res = [children_res[i] for i, _ in enumerate(node.fields)]
        len_fields = len(node.fields)
        type_parameters_res = [children_res[i + len_fields] for i, _ in enumerate(node.type_parameters)]
        res = "{indent}struct {name}".format(
            indent=" "*old_indent,
            name=node.name,
        )
        if type_parameters_res:
            res += "<" + ", ".join(type_parameters_res) + ">"
        res += " {\n"
        if fields_res:
            res += ",\n".join(fields_res)
            res += ",\n"
        res += " "*old_indent + "}"
        self.indent = old_indent
        return res

    @append_to
    def visit_field_decl(self, node):
        return "{indent}{name}: {type}".format(
            indent=" "*self.indent,
            name=node.name,
            type=self.get_type_name(node.field_type)
        )

    @append_to
    def visit_struct_instantiation(self, node):
        old_indent = self.indent
        self.indent = 0
        children = node.children()
        for c in children:
            c.accept(self)
        self.indent = old_indent
        children_res = self.pop_children_res(children)
        type_args = (
            "::<" + ", ".join([self.get_type_name(t) for t in node.stype.type_args]) + ">"
            if node.stype.is_parameterized() and node.stype.type_args
            else ""
        )
        field_names = node.field_names
        res = "{indent}{name}{type_args} {{ ".format(
            indent=" "*self.indent,
            name=node.struct_name,
            type_args=type_args
        )
        for ind, field_name in enumerate(field_names):
            res += field_name + ": " + children_res[ind] + ", "
        res += "}"
        return res

    @append_to
    def visit_field_access(self, node):
        old_indent = self.indent
        self.indent = 0
        children = node.children()
        for c in children:
            c.accept(self)
        children_res = self.pop_children_res(children)
        self.indent = old_indent
        if children:
            receiver_expr = (
                '({}).'.format(children_res[0])
                if isinstance(node.expr, ast.BottomConstant) or
                   isinstance(node.expr, ast.StructInstantiation)
                else children_res[0] + '.'
            )
        else:
            receiver_expr = ''
        res = "{indent}{receiver}{name}".format(
            indent=" " * old_indent,
            receiver=receiver_expr,
            name=node.field
        )
        #function call on struct field requires bracketing
        self.indent = old_indent
        return res
    
    @append_to
    def visit_impl(self, node):
        old_indent = self.indent
        self.indent += 2
        children = node.children()
        for c in children:
            c.accept(self)
        children_res = self.pop_children_res(children)
        functions_res = [children_res[i] for i, _ in enumerate(node.functions)]
        len_functions = len(node.functions)
        type_parameters_res = [children_res[i + len_functions] for i, _ in enumerate(node.type_parameters)]
        res = "{indent}impl{params} {trait} for {struct} {{\n".format(
            indent = " "*old_indent,
            params = "<" + ", ".join(type_parameters_res) + ">" if type_parameters_res else "",
            trait = self.get_type_name(node.trait), #??? check if correct
            struct = self.get_type_name(node.struct) #??? check if correct
        )
        if functions_res:
            res += "\n".join(functions_res)
            res += "\n"
        res += " "*old_indent + "}"
        self.indent = old_indent
        return res
