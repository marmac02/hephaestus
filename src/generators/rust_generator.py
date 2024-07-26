"""
This file includes the program generator for Rust
"""
from src.generators.generator import Generator
import functools
from collections import defaultdict
from copy import deepcopy
from typing import Tuple, List, Callable, Dict

from src import utils as ut
from src.generators import generators as gens
from src.generators import utils as gu
from src.generators.config import cfg
from src.ir import ast, types as tp, type_utils as tu, rust_types as rt
from src.ir.context import Context
from src.ir.builtins import BuiltinFactory
from src.ir import BUILTIN_FACTORIES
from src.modules.logging import Logger, log

class RustGenerator(Generator):
    def __init__(self,
                 language=None,
                 options={},
                 logger=None,
                 seed=None):
        assert language is not None, "You must specify the language"
        self.language = language
        self.logger: Logger = logger
        self.context: Context = None
        self.bt_factory: BuiltinFactory = BUILTIN_FACTORIES[language]
        self.depth = 1
        self._vars_in_context = defaultdict(lambda: 0)
        self.namespace = ('global',)

        #maps impl block ids to available field variables
        self._field_vars = {}

        #flag to handle move semantics in Rust
        self.move_semantics = False

        #Map describing impl blocks. It maps struct_names to tuples
        self._impls = {}

        #This flag is used for Rust inner functions
        self._inside_inner_function = False

        #This flag is used for Rust to handle function calls in inner functions inside trait declarations
        self._inside_trait_decl = False
        
        self.function_type = type(self.bt_factory.get_function_type())
        self.function_types = self.bt_factory.get_function_types(
            cfg.limits.max_functional_params)
        self.ret_builtin_types = self.bt_factory.get_non_nothing_types()
        self.builtin_types = self.ret_builtin_types + \
            [self.bt_factory.get_void_type()]
        self.fallback_map = {}
        # In some case we need to use two namespaces. One for having access
        # to variables from scope, and one for adding new declarations.
        # In most cases one namespace is enough, but in cases like
        # `generate_expr` we need both namespaces.
        # To use one namespace we must model scope better.
        # Almost always declaration_namespace is set to None to be ignored
        self.declaration_namespace = None
        self.int_stream = iter(range(1, 10000))

        self._blacklisted_traits: set = set()
        self._blacklisted_structs: set = set()

    ### Entry Point Generators ###

    def generate(self, context=None) -> ast.Program:
        """Generate a program.
        
        It first generates a number `n` top-level declarations,
        and then it generates the main function.
        """
        self.context = context or Context()
        for _ in ut.random.range(cfg.limits.min_top_level,
                                 cfg.limits.max_top_level):
            self.gen_top_level_declaration()
        self.generate_main_func()
        return ast.Program(self.context, self.language)

    def gen_top_level_declaration(self):
        """Generate a top-level declaration and add it in the context.

        Top-level declarations are defined in the global scope.
        Top-level declarations can be:

        * Static Variable Declarations
        * Function Declarations
        * Struct Declarations
        * Trait Declarations
        * Impl Blocks

        NOTE that a top-level declaration can generate more top-level
        declarations.
        """
        candidates = [
            self.gen_global_variable_decl,
            self.gen_func_decl,
            self.gen_struct_decl,
            self.gen_trait_decl,
            self.gen_impl,
        ]
        gen_func = ut.random.choice(candidates)
        gen_func()

    def generate_main_func(self) -> ast.FunctionDeclaration:
        """Generate the main function.
        """
        initial_namespace = self.namespace
        self.namespace += ('main', )
        initial_depth = self.depth
        self.depth += 1
        main_func = ast.FunctionDeclaration(
            "main",
            params=[],
            ret_type=self.bt_factory.get_void_type(),
            body=None,
            func_type=ast.FunctionDeclaration.FUNCTION)
        self._add_node_to_parent(ast.GLOBAL_NAMESPACE, main_func)
        expr = self.generate_expr()
        decls = list(self.context.get_declarations(
            self.namespace, True).values())
        decls = [d for d in decls
                 if not isinstance(d, ast.ParameterDeclaration)]
        body = ast.Block(decls + [expr])
        main_func.body = body
        self.depth = initial_depth
        self.namespace = initial_namespace
        return main_func


    ### Type utils adjusted for Rust ###
    def instantiate_type_constructor(self, t : tp.Type, 
                                     types_list : List[tp.Type], 
                                     type_var_map=None):
        """Instantiate a type constructor with random type arguments.
           Calls the instantiate_type_constructor function in the type_utils module,
           and concretizes trait types.
        """
        inst_t, param_map = tu.instantiate_type_constructor(t, types_list, type_var_map=type_var_map, disable_variance=True)
        param_map = self._concretize_map(param_map)
        inst_t, _ = tu.instantiate_type_constructor(t, types_list, type_var_map=param_map, disable_variance=True)
        return inst_t, param_map

    def instantiate_parameterized_function(self, 
                                           type_params : List[tp.TypeParameter], 
                                           types_list : List[tp.Type],
                                           type_var_map=None):
        """Instantiate a parameterized function with random type arguments.
           Calls the instantiate_parameterized_function function in the type_utils module,
           and concretizes trait types.
        """
        param_map = tu.instantiate_parameterized_function(type_params, types_list, type_var_map=type_var_map, only_regular=True)
        param_map = self._concretize_map(param_map)
        return param_map
    
    def unify_types(self, t1 : tp.Type, t2 : tp.Type, visited_counter=None):
        """ Unifies two types and returns a type map if unification is successful.
            Calls the unify_types function in the type_utils module.
        """
        type_map = tu.unify_types(t1, self._erase_bounds(t2), self.bt_factory, same_type=False)
        if type_map:
            self._restore_bounds(t2, type_map)
        visited_counter = visited_counter or defaultdict(int)
        if self._map_fulfills_bounds(type_map, visited_counter):
            return type_map
        return {}

    def _erase_bounds(self, t):
        """ Erase type bounds from a type
        """
        updated_type = deepcopy(t)
        if t.is_type_var():
            updated_type.bound = None
        if t.is_parameterized():
            updated_type_args = []
            for t_arg in t.type_args:
                updated_type_args.append(self._erase_bounds(t_arg))
            updated_type.type_args = updated_type_args
            return updated_type
        return updated_type

    def _restore_bounds(self, t, type_map):
        """ Restore bounds in a type map
            t is a type containing type parameters with bounds
            type_map maps type parameters stripped of bounds to instantiaions
        """
        if t.is_type_var() and self._erase_bounds(t) in type_map.keys():
            assigned_inst = type_map[self._erase_bounds(t)]
            type_map.pop(self._erase_bounds(t))
            type_map[t] = assigned_inst
        if t.is_parameterized():
            for t_arg in t.type_args:
                self._restore_bounds(t_arg, type_map)

    def _map_fulfills_bounds(self, type_map, visited_counter):
        """ Check if the type map fulfills the bounds of the type parameters """
        def is_struct_compatible(impl_struct, t, trait_map):
            if not impl_struct.has_type_variables():
                return impl_struct == t
            else:
                struct_map = self.unify_types(t, impl_struct)
                if not struct_map:
                    return False
                for (t_param, type_inst) in struct_map.items():
                    if t_param in trait_map.keys() and trait_map[t_param] != type_inst:
                        return False
                return True
        
        for t_param, t in type_map.items():
            if t_param.bound is None:
                continue
            trait_type = t_param.bound
            for (impl, _, _) in sum(self._impls.values(), []):
                if visited_counter[impl.name] > cfg.limits.trait.max_concretization_depth:
                    continue
                if not impl.trait.has_type_variables():
                    if impl.trait == trait_type and is_struct_compatible(impl.struct, t, {}):
                        return True
                else:
                    visited_counter[impl.name] += 1
                    trait_map = self.unify_types(trait_type, impl.trait, visited_counter)
                    if not trait_map:
                        continue
                    if is_struct_compatible(impl.struct, t, trait_map):
                        return True
            #No matching impl found for the type parameter
            return False
        return True

    def _concretize_map(self, type_map):
        """ Replace abstract trait type parameter instantiation with a
            matching concrete type implementing this trait
        """
        if type_map is None:
            return None
        updated_map = {}
        prev_type_map = {}
        for key in type_map.keys():
            if key.is_type_var() and key.bound is not None:
                updated_map[key] = self.concretize_type(type_map[key], prev_type_map, defaultdict(int))
            else:
                updated_map[key] = type_map[key]
        return updated_map
    
    def concretize_type(self, t, prev_type_map=None, visited_counter=None):
        """ Replace abstract trait type with a matching concrete type implementing this trait.
            By design of creating trait bounds, there should exist a matching struct declaration.
            Args:
                t: type to be concretized
                prev_type_map: maps previously concretized types to their instantiations
                    Needed for cases when a type parameter bound depends on other type parameters
                    from the same list, i.e. Foo< A : T1, B : T2<A> >
                visited_counter: a dictionary to keep track of the depth of type instantiation
                    We need this to avoid infinite recursion when unifying trait types - this breaks
                    infinite cycles by forbidding using the same implementation more than
                    self.instantiation_depth times
            Returns:
                concretized type t, that is some type implementing the trait type t,
                or t if it is already a concrete type
        """
        if t in prev_type_map:
            #type has been concretized already
            return prev_type_map[t]
        if t.name in self.context.get_traits(self.namespace).keys():
            #type is a trait type, it must be concretized
            trait_type = deepcopy(t)
            if t.is_parameterized():
                trait_type.type_args = [self.concretize_type(t_arg, prev_type_map, visited_counter) for t_arg in trait_type.type_args]
            impl_list = sum(self._impls.values(), [])
            for (impl, _, _) in impl_list:
                if impl.trait.name != trait_type.name or visited_counter[impl.name] > cfg.limits.trait.max_concretization_depth:
                    #Impl is not for the trait_type
                    continue
                visited_counter[impl.name] += 1
                impl_type_map = {t_param: self.select_type() 
                                if t_param.bound is None 
                                else self.concretize_type(t_param.bound, prev_type_map, visited_counter)
                                for t_param in impl.type_parameters
                            }
                visited_counter[impl.name] -= 1
                if not impl.trait.has_type_variables():
                    if impl.trait == trait_type:
                        updated_type = tp.substitute_type(impl.struct, impl_type_map)
                        prev_type_map[t] = updated_type
                        return updated_type
                else:
                    trait_map = self.unify_types(trait_type, impl.trait)
                    if not trait_map:
                        continue
                    impl_type_map.update(trait_map)
                    updated_type = tp.substitute_type(impl.struct, impl_type_map)
                    prev_type_map[t] = updated_type
                    return updated_type
            #If we reach this point, there is a cycle in the impl declarations
            #Break the cycle by creating a new type for the trait t
            if t in self.fallback_map:
                return self.fallback_map[t]
            initial_namespace = self.namespace
            self.namespace = ast.GLOBAL_NAMESPACE
            fallback_struct = self.gen_struct_decl(init_type_params=[])
            self.gen_impl(struct_trait_pair=(fallback_struct.get_type(), t))
            self.namespace = initial_namespace
            self.fallback_map[t] = fallback_struct.get_type()
            return fallback_struct.get_type()
        if t.name in [fn_trait.name for fn_trait in self.bt_factory.get_fn_trait_classes()]:
            #type is a function trait type, it must be concretized
            func_constr = self.bt_factory.get_function_type(len(t.type_args) - 1)
            concrete_fun_type = func_constr.new(t.type_args)
            return concrete_fun_type
        if t.is_parameterized():
            #type is a parameterized type, its type args must be concretized
            updated_type_args = []
            for t_arg in t.type_args:
                updated_type_args.append(self.concretize_type(t_arg, prev_type_map, visited_counter))
            updated_type = deepcopy(t)
            updated_type.type_args = updated_type_args
            prev_type_map[t] = updated_type
            return updated_type
        return t


    ### Generators ###

    ##### Declarations #####

    def _get_decl_from_var(self, var):
        var_decls = self.context.get_vars(self.namespace)
        if var.name in var_decls.keys():
            return var_decls[var.name]
        for fv in self._get_field_vars():
            if fv.name == var.name:
                return fv
        raise ValueError("Variable declaration cannot be found")

    def _remove_unused_type_params(self, type_params, types_list):
        """
        Remove type parameters that do not appear in the types_list.
        types_list is e.g. a list of types that are used in the function signature,
        or in the struct declaration.
        """
        def get_type_vars(t):
            if t.is_type_var():
                return [t]
            return getattr(t, "get_type_variables", lambda x: [])(
                self.bt_factory
            )
        all_type_vars = []
        for t in types_list:
            all_type_vars.extend(get_type_vars(t))
        for t_param in type_params:
            if t_param not in all_type_vars:
                self.context.remove_type(self.namespace, t_param.name)
        remap = {}
        for type_param in type_params:
            if type_param not in all_type_vars:
                if type_param.bound is not None:
                    type_param.bound = tp.substitute_type(type_param.bound, remap)
                remap[type_param] = self.select_type() \
                                    if type_param.bound is None \
                                    else self.concretize_type(type_param.bound, {}, defaultdict(int))
        
        type_params = [t_param for t_param in type_params if t_param in all_type_vars]
        for t_param in type_params:
            if t_param.bound is not None:
                t_param.bound = tp.substitute_type(t_param.bound, remap)
        return type_params

    def _add_used_type_params(self, type_params, params, ret_type):
        """
        Add function's type parameters that are included in its signature, 
        and are not defined in trait declaration if function is a trait function.
        """
        def get_type_vars(t):
            if t.is_type_var():
                return [t]
            return getattr(t, "get_type_variables", lambda x: [])(
                self.bt_factory
            )
        outer_trait_type_params = self._get_outer_trait_type_params() #type params for outer trait if any
        all_type_vars = []
        param_types = [p.get_type() for p in params]
        for t in param_types + [ret_type]:
            all_type_vars.extend(get_type_vars(t))
        for t_param in all_type_vars:
            if t_param not in type_params and t_param not in outer_trait_type_params:
                type_params.append(t_param)
        return type_params

    def _get_outer_trait_type_params(self):
        """ Function returns type parameters of the trait in whose namespace
            the function is declared.
        """
        trait_type_params = []
        ns = self.namespace
        if len(ns) > 2 and ns[-2][0].isupper():
            trait_name = ns[-2]
            trait_decl = self.context.get_traits(self.namespace)[trait_name]
            return trait_decl.type_parameters
        return []

    def gen_func_decl(self,
                      etype:tp.Type=None,
                      not_void=False,
                      func_name:str=None,
                      params:List[ast.ParameterDeclaration]=None,
                      is_signature=False,
                      trait_func=False,
                      type_params:List[tp.TypeParameter]=None,
                      namespace=None) -> ast.FunctionDeclaration:
        """Generate a function declaration.

        This method is responsible for generating all types of function/methods,
        i.e. functions, trait methods, nested functions. Furthermore, it also
        generates parameterized functions.

        Args:
            etype: expected return type.
            not_void: do not return void.
            func_name: function name.
            params: list of parameter declarations.
            is_signature: function is a signature (without body).
            type_params: list of type parameters for parameterized function.
            namespace: set explicit namespace.

        Returns:
            A function declaration node.
        """
        func_name = func_name or gu.gen_identifier('lower')
        initial_namespace = self.namespace
        if namespace:
            self.namespace = namespace + (func_name,)
        else:
            self.namespace += (func_name,)
        initial_depth = self.depth
        self.depth += 1

        #check if function is declared in a trait
        trait_func = self.namespace[-2][0].isupper() 
        # Check if this function we want to generate is a nested functions.
        # To do so, we want to find if the function is directly inside the
        # namespace of another function.
        nested_function = (len(self.namespace) > 1 and
                           self.namespace[-2] != 'global' and
                           self.namespace[-2][0].islower())

        prev_inside_inner_function = self._inside_inner_function
        self._inside_inner_function = nested_function

        if (not nested_function and not trait_func):
            if type_params is not None:
                for t_p in type_params:
                    # We add the types to the context.
                    self.context.add_type(self.namespace, t_p.name, t_p)
            else:
                type_params = self.gen_type_params(
                    blacklist=self._get_type_variable_names(),
                    for_function=True
                ) if ut.random.bool(prob=cfg.prob.parameterized_functions) \
                  else []

        else:
            type_params = []
        if params is not None:
            for p in params:
                log(self.logger, "Adding parameter {} to context in gen_func_decl".format(p.name))
                self._add_node_to_parent(self.namespace, p)
        else:
            params = self._gen_func_params()
        ret_type = self._get_func_ret_type(params, etype, not_void=not_void)
        if is_signature:
            body = None
        else:
            body = ast.BottomConstant(ret_type)
        type_params = self._remove_unused_type_params(type_params, [p.get_type() for p in params] + [ret_type])
        type_params = self._add_used_type_params(type_params, params, ret_type)
        if trait_func:
            #adding &self parameter for Rust trait methods
            params = [ast.SelfParameter()] + params
        func = ast.FunctionDeclaration(
            func_name, params, ret_type, body,
            func_type=1,
            inferred_type=None,
            type_parameters=type_params,
            trait_func=trait_func,
        )
        msg = "Adding function to context {} {}".format(func_name, ", ".join([str(p) for p in params]))
        log(self.logger, msg)
        self._add_node_to_parent(self.namespace[:-1], func)
        for p in params:
            if isinstance(p, ast.SelfParameter):
                continue
            self.context.add_var(self.namespace, p.name, p)
            msg = "Adding parameter to context: {} in function {}".format(p.name, func_name)
            log(self.logger, msg)
        if func.body is not None:
            body = self._gen_func_body(ret_type)
        func.body = body
        self._inside_inner_function = prev_inside_inner_function
        self.depth = initial_depth
        self.namespace = initial_namespace
        return func


    def _add_node_to_struct(self, struct, node):
        if isinstance(node, ast.FieldDeclaration):
            struct.fields.append(node)
            return
        assert False, ('Trying to put a node in struct other than a field')

    def _add_node_to_trait(self, trait, node):
        if isinstance(node, ast.FunctionDeclaration):
            if node.body is None:
                trait.function_signatures.append(node)
            else:
                trait.default_impls.append(node)

    
    def _add_node_to_parent(self, parent_namespace, node):
        node_type = {
            ast.FunctionDeclaration: self.context.add_func,
            ast.VariableDeclaration: self.context.add_var,
            ast.FieldDeclaration: self.context.add_var,
            ast.ParameterDeclaration: self.context.add_var,
            ast.Lambda: self.context.add_lambda,
            ast.TraitDeclaration: self.context.add_trait,
            ast.StructDeclaration: self.context.add_struct,
            ast.Impl: self.context.add_impl,
        }
        if parent_namespace == ast.GLOBAL_NAMESPACE:
            node_type[type(node)](parent_namespace, node.name, node)
            return
        parent = self.context.get_decl(parent_namespace[:-1],
                                       parent_namespace[-1])
        if parent:
            if isinstance(parent, ast.StructDeclaration):
                self._add_node_to_struct(parent, node)
            if isinstance(parent, ast.TraitDeclaration):
                self._add_node_to_trait(parent, node)

        node_type[type(node)](parent_namespace, node.name, node)

    def gen_global_variable_decl(self) -> ast.VariableDeclaration:
        """Generate a global Variable Declaration in Rust.
           Global (static) variable declarations are final, 
           and cannot contain function calls.
        """
        available_types = [self.bt_factory.get_integer_type(),
                           self.bt_factory.get_float_type(),
                           self.bt_factory.get_boolean_type(),
                           self.bt_factory.get_char_type()]
        var_type = ut.random.choice(available_types)
        initial_depth = self.depth
        self.depth += 1
        expr = self.generate_expr(var_type, True)
        self.depth = initial_depth
        var_decl = ast.VariableDeclaration(gu.gen_identifier('lower'),
                                           expr=expr,
                                           is_final=True,
                                           var_type=var_type,
                                           inferred_type=var_type)
        log(self.logger, "Adding global variable {} to context".format(var_decl.name))
        log(self.logger, "Namespace of the global variable: {}".format(self.namespace))
        self._add_node_to_parent('global', var_decl)
        return var_decl

    def gen_variable_decl(self,
                          etype=None,
                          only_leaves=False,
                          expr=None) -> ast.VariableDeclaration:
        """Generate a Variable Declaration.

        Args:
            etype: the type of the variable.
            only_leaves: do not generate new leaves except from `expr`.
            expr: an expression to assign to the variable.

        Returns:
            A Variable Declaration
        """
        prev_move_semantics = self.move_semantics
        self.move_semantics = True
        var_type = etype if etype else self.select_type()
        initial_depth = self.depth
        self.depth += 1
        expr = expr or self.generate_expr(var_type, only_leaves,
                                          sam_coercion=True)
        if isinstance(expr, ast.Variable):
            vard = self._get_decl_from_var(expr)
            vard.is_moved = self._move_condition(vard)
            vard.move_prohibited = self._type_moveable(vard)
        self.depth = initial_depth
        is_final = ut.random.bool()
        vtype = var_type.get_bound_rec() if var_type.is_wildcard() else \
            var_type
        var_decl = ast.VariableDeclaration(
            gu.gen_identifier('lower'),
            expr=expr,
            is_final=is_final,
            var_type=vtype,
            inferred_type=var_type)
        log(self.logger, "Adding variable {} to context in gen_variable_decl".format(var_decl.name))
        self._add_node_to_parent(self.namespace, var_decl)
        self.move_semantics = prev_move_semantics
        return var_decl

    ##### Expressions #####

    def _type_moveable(self, decl) -> bool:
        """ Check if type is subject to move semantics rules """
        return not decl.get_type().is_primitive() and not decl.get_type().is_function_type()


    def generate_expr(self,
                      expr_type: tp.Type=None,
                      only_leaves=False,
                      subtype=True,
                      exclude_var=False,
                      gen_bottom=False,
                      sam_coercion=False) -> ast.Expr:
        """Generate an expression.

        This function could produce new nodes external to the generated
        expression as a side effect. For instance, it could generate new
        variable declarations.

        Args:
            expr_type: The type that the expression should have.
            only_leaves: do not generate new leaves except from `expr`.
            subtype: The type of the generated expression could be a subtype
                of `expr_type`.
            exclude_var: if this option is false, then it could assign the
                generated expression into a variable, and return that
                variable reference.
            gen_bottom: Generate a bottom constant.
            sam_coercion: Enable sam coercion.

        Returns:
            The generated expression.
        """
        if gen_bottom:
            return ast.BottomConstant(None)
        expr_type = expr_type or self.select_type()
        gen_var = (
            not only_leaves and
            expr_type != self.bt_factory.get_void_type() and
            self._vars_in_context[self.namespace] < cfg.limits.max_var_decls and
            ut.random.bool()
        )
        generators = self.get_generators(expr_type, only_leaves, expr_type,
                                         exclude_var, sam_coercion=sam_coercion)
        prev_move_semantics = self.move_semantics
        self.move_semantics = True if gen_var else self.move_semantics
        expr = ut.random.choice(generators)(expr_type)
        self.move_semantics = prev_move_semantics
        # Make a probabilistic choice, and assign the generated expr
        # into a variable, and return that variable reference.
        if gen_var:
            self._vars_in_context[self.namespace] += 1
            var_decl = self.gen_variable_decl(expr_type, only_leaves, expr=expr)
            var_decl.is_moved = self._type_moveable(var_decl)
            var_decl.move_prohibited = self._type_moveable(var_decl)
            expr = ast.Variable(var_decl.name)
        return expr

    # pylint: disable=unused-argument
    def gen_assignment(self,
                       expr_type: tp.Type,
                       only_leaves=False,
                       subtype=True) -> ast.Assignment:
        """Generate an assignment expressions.

        Args:
            expr_type: The value that the assignment expression should hold.
            only_leaves: do not generate new leaves except from `expr`.
            subtype: The type of the generated expression could be a subtype
                of `expr_type`.
        """
        prev_move_semantics = self.move_semantics
        self.move_semantics = True
        # Get all non-final variables for performing the assignment.
        only_current_namespace = self._inside_inner_function
        variables = self._get_assignable_vars(only_current_namespace)
        initial_depth = self.depth
        self.depth += 1
        if not variables:
            # Nothing of the above worked, so generate a 'var' variable,
            # and perform the assignment
            etype = self.select_type(exclude_covariants=True,
                                     exclude_contravariants=True)
            self._vars_in_context[self.namespace] += 1
            # If there are not variable declarations that match our criteria,
            # we have to create a new variable declaration.
            var_decl = self.gen_variable_decl(etype, only_leaves)
            var_decl.is_final = False
            var_decl.var_type = var_decl.get_type()
            self.depth = initial_depth
            assignment = ast.Assignment(var_decl.name,
                                  self.generate_expr(var_decl.get_type(),
                                                     only_leaves, subtype))
            self.move_semantics = prev_move_semantics
            return assignment
        receiver, variable = ut.random.choice(variables)
        self.depth = initial_depth
        gen_bottom = (
            variable.get_type().is_wildcard() or
            (
                variable.get_type().is_parameterized() and
                variable.get_type().has_wildcards()
            )
        )
        right_side = self.generate_expr(variable.get_type(), only_leaves, subtype, gen_bottom=gen_bottom)
        self.move_semantics = prev_move_semantics
        return ast.Assignment(variable.name, right_side,
                              receiver=receiver,)

    # Where

    def _get_assignable_vars(self, only_current_namespace) -> List[ast.Variable]:
        """Get all non-final variables in context.
        """
        variables = []
        for var in self.context.get_vars(namespace=self.namespace, only_current=only_current_namespace).values():
            if getattr(var, 'is_final', True):
                continue
            variables.append((None, var))
            if var.is_moved:
                continue
            if var.get_type().name in self.context.get_structs(self.namespace).keys():
                type_map = {}
                if var.get_type().is_parameterized():
                    type_map = dict(zip(var.get_type().t_constructor.type_parameters, var.get_type().type_args))
                struct_decl = self.context.get_structs(self.namespace)[var.get_type().name]
                for f in struct_decl.fields:
                    updated_type = tp.substitute_type(f.get_type(), type_map)
                    updated_field_decl = ast.FieldDeclaration(f.name, updated_type)
                    variables.append((ast.Variable(var.name), updated_field_decl))
        return variables


    def gen_field_access(self,
                         etype: tp.Type,
                         only_leaves=False,
                         subtype=True) -> ast.FieldAccess:
        """Generate a field access expression.

        Args:
            expr_type: The type of the value that the field access should return.
            only_leaves: do not generate new leaves except from `expr`.
            subtype: The type of the generated expression could be a subtype
                of `expr_type`.
        """
        log(self.logger, "Generating field access with type {}".format(etype))
        initial_depth = self.depth
        self.depth += 1
        objs = self._get_matching_objects(etype, subtype, 'fields')
        if not objs:
            type_f = self.get_matching_struct(etype, subtype=subtype)
            if type_f is None:
                type_f = self._gen_matching_struct(
                    etype, not_void=True,
                )
            receiver = self.generate_expr(type_f.receiver_t, only_leaves)
            objs.append(gu.AttrReceiverInfo(
                receiver, None, type_f.attr_decl, None))
        objs = [(obj.receiver_expr, obj.attr_decl) for obj in objs]
        receiver, attr = ut.random.choice(objs)
        if isinstance(receiver, ast.Variable):
            #prevent use of variables that were partially moved
            var_decl = self._get_decl_from_var(receiver)
            var_decl.is_moved = self.move_semantics
            var_decl.move_prohibited = self._type_moveable(var_decl)
        self.depth = initial_depth
        return ast.FieldAccess(receiver, attr.name)

    def _gen_matching_struct(self,
                             etype: tp.Type,
                             not_void=False,
                             signature=False) -> gu.AttrAccessInfo:
        """ Generate a struct that has a field that is etype
        """
        initial_namespace = self.namespace
        struct_name = gu.gen_identifier('capitalize')
        type_params = None
        if etype.has_type_variables():
            self.namespace = ast.GLOBAL_NAMESPACE + (struct_name,)
            type_params, type_var_map, can_wildcard = self._create_type_params_from_etype(etype)
            etype2 = tp.substitute_type(etype, type_var_map)
        else:
            type_var_map, etype2, can_wildcard = {}, etype, False
        self.namespace = ast.GLOBAL_NAMESPACE
        struct_type_map = None
        
        s = self.gen_struct_decl(struct_name=struct_name, field_type=etype2, init_type_params=type_params)
        
        self.namespace = initial_namespace
        if s.is_parameterized():
            type_map = {v: k for k, v in type_var_map.items()} 
            if etype2.is_primitive() and (
                    etype2 == self.bt_factory.get_void_type()):
                type_map = None
            s_type, params_map = self.instantiate_type_constructor(s.get_type(), self.get_types(), type_map)
        else:
            s_type, params_map = s.get_type(), {}
        for attr in self._get_struct_attributes(s, 'fields'):
            if not self._is_sigtype_compatible(attr, etype, params_map, signature, False):
                continue
            return gu.AttrAccessInfo(s_type, params_map, attr, {})
        return None

    def get_matching_struct(self,
                            etype: tp.Type,
                            subtype: bool) -> gu.AttrAccessInfo:
        """ Get a struct that has a field of type etype. """
        struct_decls = self._get_matching_struct_decls(etype, subtype)
        if not struct_decls:
            return None
        s, type_var_map, attr = ut.random.choice(struct_decls)
        func_type_var_map = {}
        is_parameterized_func = isinstance(
            attr, ast.FunctionDeclaration) and attr.is_parameterized()
        if s.is_parameterized():
            s_type, params_map = self.instantiate_type_constructor(s.get_type(), self.get_types(), type_var_map)
        else:
            s_type, params_map = s.get_type(), {}
        return gu.AttrAccessInfo(s_type, params_map, attr, func_type_var_map)

    def _get_matching_struct_decls(self,
                                 etype: tp.Type,
                                 subtype: bool,
                                 signature=False) -> List[Tuple[ast.StructDeclaration,
                                                          tu.TypeVarMap,
                                                          ast.Declaration]]:
        """Get structs that have fields of type etype."""
        struct_decls = []
        for s in self.context.get_structs(self.namespace).values():
            for attr in self._get_struct_attributes(s, 'fields'):
                attr_type = attr.get_type()
                if not attr_type:
                    continue
                if attr_type == self.bt_factory.get_void_type():
                    continue
                if attr.name == self.namespace[-1] and signature:
                    continue
                is_comb, type_var_map = self._is_signature_compatible(
                    attr, etype, signature, subtype)
                if not is_comb:
                    continue
                struct_decls.append((s, type_var_map, attr))
        return struct_decls
    

    def gen_variable(self,
                     etype: tp.Type,
                     only_leaves=False,
                     subtype=True) -> ast.Variable:
        """Generate a variable.

        First, it searches for all variables in the scope. In case it doesn't
        find any variable of etype, then it generates one.

        Args:
            expr_type: The type that the variable should have.
            only_leaves: do not generate new leaves except from `expr`.
            subtype: The type of the generated variable could be a subtype
                of `expr_type`.
        """
        # Get all variables declared in the current namespace or
        # the outer namespace.
        variables = self.context.get_vars(self.namespace).values()
        if self._inside_inner_function:
            variables = list(self.context.get_vars(namespace=self.namespace, only_current=True).values())
        else:
            variables = list(variables) + self._get_field_vars()
        variables = [v for v in variables if (not v.is_moved) and (not self.move_semantics or not v.move_prohibited)]
        # If we need to use a variable of a specific types, then filter
        # all variables that match this specific type.
        if subtype:
            fun = lambda v, t: v.get_type().is_assignable(t)
        else:
            fun = lambda v, t: v.get_type() == t
        variables = [v for v in variables if fun(v, etype)]
        if not variables:
            return self.generate_expr(etype, only_leaves=only_leaves,
                                      subtype=subtype, exclude_var=True)
        varia = ut.random.choice([v for v in variables])
        varia.is_moved = self._move_condition(varia)
        varia.move_prohibited = True
        return ast.Variable(varia.name)

    def _move_condition(self, varia):
        """ Checks if variable is moved
        """
        if not varia.get_type().is_primitive() and not varia.get_type().is_function_type() and self.move_semantics:
            return True
        return False

    def _get_field_vars(self) -> List[ast.VariableDeclaration]:
        """ Get all field variables accessible in the current impl block. """
        for ns in self.namespace:
            if ns.startswith('impl'):
                if ns in self._field_vars.keys():
                    #if field var is not primitive, and would be used in a move-inducing operation, it should be excluded
                    field_vars = [v for v in self._field_vars[ns] if (v.get_type().is_primitive() or not self.move_semantics)]
                    return field_vars
        return []

    def gen_array_expr(self,
                       expr_type: tp.Type,
                       only_leaves=False,
                       subtype=True) -> ast.ArrayExpr:
        """Generate an array expression

        Args:
            expr_type: The type of the array
            only_leaves: do not generate new leaves except from `expr`.
            subtype: The type of the generated array could be a subtype
                of `expr_type`.
        """
        prev_move_semantics = self.move_semantics
        self.move_semantics = True
        arr_len = ut.random.integer(0, 3)
        etype = expr_type.type_args[0]
        exprs = [
            self.generate_expr(etype, only_leaves=only_leaves, subtype=subtype)
            for _ in range(arr_len)
        ]
        self.move_semantics = prev_move_semantics
        return ast.ArrayExpr(expr_type.to_variance_free(), arr_len, exprs)

    # pylint: disable=unused-argument
    def gen_equality_expr(self,
                          expr_type=None,
                          only_leaves=False) -> ast.EqualityExpr:
        """Generate an equality expression

        It generates two additional expression for performing the comparison
        between them.

        Args:
            expr_type: exists for compatibility reasons.
            only_leaves: do not generate new leaves except from `expr`.
        """
        prev_move_semantics = self.move_semantics
        self.move_semantics = False
        initial_depth = self.depth
        self.depth += 1
        etype = self.select_type(exclude_function_types=True, exclude_type_vars=True, exclude_usr_types=True)
        op = ut.random.choice(ast.EqualityExpr.VALID_OPERATORS[self.language])
        e1 = self.generate_expr(etype, only_leaves, subtype=False)
        e2 = self.generate_expr(etype, only_leaves, subtype=False)
        self.depth = initial_depth
        self.move_semantics = prev_move_semantics
        return ast.EqualityExpr(e1, e2, op)

    def gen_comparison_expr(self,
                            expr_type=None,
                            only_leaves=False) -> ast.ComparisonExpr:
        """Generate a comparison expression

        It generates two additional expression for performing the comparison
        between them.
        It supports only built-in types.

        Args:
            expr_type: exists for compatibility reasons.
            only_leaves: do not generate new leaves except from `expr`.
        """
        prev_move_semantics = self.move_semantics
        self.move_semantics = False
        valid_types = [
            self.bt_factory.get_string_type(),
            self.bt_factory.get_boolean_type(),
            self.bt_factory.get_double_type(),
            self.bt_factory.get_char_type(),
            self.bt_factory.get_float_type(),
            self.bt_factory.get_integer_type(),
            self.bt_factory.get_byte_type(),
            self.bt_factory.get_short_type(),
            self.bt_factory.get_long_type(),
            self.bt_factory.get_big_decimal_type(),
            self.bt_factory.get_big_integer_type(),
        ]
        number_types = self.bt_factory.get_number_types()
        e2_types = {
            self.bt_factory.get_string_type(): [
                self.bt_factory.get_string_type()
            ],
            self.bt_factory.get_boolean_type(): [
                self.bt_factory.get_boolean_type()
            ],
            self.bt_factory.get_double_type(): [
                self.bt_factory.get_double_type()
            ],
            self.bt_factory.get_big_decimal_type(): [
                self.bt_factory.get_big_decimal_type()
            ],
            self.bt_factory.get_char_type(): [
                self.bt_factory.get_char_type()
            ],
            self.bt_factory.get_float_type(): [
                self.bt_factory.get_float_type()
            ],
            self.bt_factory.get_integer_type(): [
                self.bt_factory.get_integer_type(),
            ],
            self.bt_factory.get_big_integer_type(): [
                self.bt_factory.get_big_integer_type()
            ],
            self.bt_factory.get_byte_type(): [
                self.bt_factory.get_byte_type()
            ],
            self.bt_factory.get_short_type(): [
                self.bt_factory.get_short_type()
            ],
            self.bt_factory.get_long_type(): [
                self.bt_factory.get_long_type()
            ]
        }
        initial_depth = self.depth
        self.depth += 1
        op = ut.random.choice(
            ast.ComparisonExpr.VALID_OPERATORS[self.language])
        e1_type = ut.random.choice(valid_types)
        e2_type = ut.random.choice(e2_types[e1_type])
        e1 = self.generate_expr(e1_type, only_leaves)
        e2 = self.generate_expr(e2_type, only_leaves)
        self.depth = initial_depth
        self.move_semantics = prev_move_semantics
        return ast.ComparisonExpr(e1, e2, op)

    def gen_conditional(self,
                        etype: tp.Type,
                        only_leaves=False,
                        subtype=True) -> ast.Conditional:
        """Generate a conditional expression.

        It generates 3 sub expressions, one for each branch, and one for
        the conditional.

        Args:
            etype: type for each sub expression.
            only_leaves: do not generate new leaves except from `expr`.
            subtype: The type of the sub expressions could be a subtype of
                `etype`.
        """
        initial_depth = self.depth
        self.depth += 3
        cond = self.generate_expr(self.bt_factory.get_boolean_type(),
                                  only_leaves)
        prev_move_semantics = self.move_semantics
        self.move_semantics = True
        true_type, false_type, cond_type = etype, etype, etype
        true_expr = ast.Block([self.generate_expr(true_type, only_leaves, subtype=False)], is_func_block = False)
        false_expr = ast.Block([self.generate_expr(false_type, only_leaves, subtype=False)], is_func_block = False)
        self.depth = initial_depth
        self.move_semantics = prev_move_semantics
        return ast.Conditional(cond, true_expr, false_expr, cond_type)


    def gen_lambda(self,
                   etype: tp.Type=None,
                   not_void=False,
                   params: List[ast.ParameterDeclaration]=None,
                   only_leaves=False
                  ) -> ast.Lambda:
        """Generate a lambda expression.

        Lambdas have shadow names that we can use them in the context to
        retrieve them.

        Args:
            etype: return type of the lambda.
            not_void: the lambda should not return void.
            shadow_name: give a specific shadow name.
            params: parameters for the lambda.
        """
        prev_inside_inner_function = self._inside_inner_function
        self._inside_inner_function = True #restricting capture of variables
        if self.declaration_namespace:
            namespace = self.declaration_namespace
        else:
            namespace = self.namespace

        initial_namespace = self.namespace
        shadow_name = "lambda_" + str(next(self.int_stream))
        self.namespace += (shadow_name,)
        initial_depth = self.depth
        self.depth += 1

        params = params if params is not None else self._gen_func_params()
        param_types = [p.param_type for p in params]
        for p in params:
            log(self.logger, "Adding parameter {} to context in gen_lambda".format(p.name))
            self.context.add_var(self.namespace, p.name, p)
        ret_type = self._get_func_ret_type(params, etype, not_void=not_void)
        signature = tp.ParameterizedType(
            self.bt_factory.get_function_type(len(params)),
            param_types + [ret_type])
        res = ast.Lambda(shadow_name, params, ret_type, None, signature)
        self.context.add_lambda(initial_namespace, shadow_name, res)
        body = self._gen_func_body(ret_type)
        res.body = body
        
        self.depth = initial_depth
        self.namespace = initial_namespace
        self._inside_inner_function = prev_inside_inner_function
        return res

    def gen_func_call(self,
                      etype: tp.Type,
                      only_leaves=False,
                      subtype=True) -> ast.FunctionCall:
        """Generate a function call.
        Args:
            etype: the type that the function call should return.
            only_leaves: do not generate new leaves except from `expr`.
            subtype: The returned type could be a subtype of `etype`.

        Returns:
            A function call.
        """
        if ut.random.bool(cfg.prob.func_ref_call):
            ref_call = self._gen_func_call_ref(etype, only_leaves, subtype)
            if ref_call:
                return ref_call
        func_call = self._gen_func_call(etype, only_leaves, subtype)
        return func_call

    # gen_func_call Where

    def _gen_func_call(self,
                       etype: tp.Type,
                       only_leaves=False,
                       subtype=True) -> ast.FunctionCall:
        """Generate a function call.

        Args:
            etype: the type that the function call should return.
            only_leaves: do not generate new leaves except from `expr`.
            subtype: The returned type could be a subtype of `etype`.
        """
        log(self.logger, "Generating function call of type {}".format(etype))
        func_decls = self._get_matching_function_declarations(etype, subtype)
        funcs = []
        #filtering functions with receivers that are moved variables
        for f in func_decls:
            if isinstance(f.receiver_expr, ast.Variable):
                var = self._get_decl_from_var(f.receiver_expr)
                if var.is_moved:
                    continue
            funcs.append(f)
        if not funcs:
            msg = "No compatible functions in the current scope for type {}"
            log(self.logger, msg.format(etype))

            type_fun = self._get_struct_with_matching_function(etype)
            
            if type_fun is None:
                msg = "No compatible structs for type {}"
                log(self.logger, msg.format(etype))
                # Here, we generate a function or a struct implementing a function
                # whose return type is 'etype'.
                type_fun = self._gen_matching_func(etype, not_void=True)
            if type_fun.receiver_t is None:
                receiver = None
            else:
                type_fun_rec = type_fun.receiver_t
                if type_fun.receiver_t.is_type_constructor():
                    type_fun_rec, _ = self.instantiate_type_constructor(type_fun.receiver_t, self.get_types(), type_fun.receiver_inst)
                receiver = self.generate_expr(type_fun_rec, only_leaves)
            funcs.append(gu.AttrReceiverInfo(receiver, type_fun.receiver_inst,
                         type_fun.attr_decl, type_fun.attr_inst))

        rand_func = ut.random.choice(funcs)
        receiver = rand_func.receiver_expr
        params_map = rand_func.receiver_inst
        func = rand_func.attr_decl
        func_type_map = rand_func.attr_inst
        params_map.update(func_type_map or {})
        
        if isinstance(receiver, ast.Variable):
            var_decl = self._get_decl_from_var(receiver)
            var_decl.move_prohibited = self._type_moveable(var_decl)

        args = []
        initial_depth = self.depth
        self.depth += 1
        prev_move_semantics = self.move_semantics
        self.move_semantics = True

        for param in func.params:
            if isinstance(param, ast.SelfParameter):
                args.append(ast.CallArgument(ast.SelfParameter()))
                continue

            expr_type = tp.substitute_type(param.get_type(), params_map)
            gen_bottom = expr_type.is_wildcard() or (
                expr_type.is_parameterized() and expr_type.has_wildcards())
            arg = self.generate_expr(expr_type, only_leaves,
                                        gen_bottom=gen_bottom)
            args.append(ast.CallArgument(arg))
        self.depth = initial_depth
        type_args = (
            []
            if not func.is_parameterized()
            else [
                func_type_map[t_param]
                for t_param in func.type_parameters
            ]
        )
        trait_func = getattr(func, 'trait_func', True)
        self.move_semantics = prev_move_semantics
        if isinstance(receiver, ast.Variable):
            var_decl = self._get_decl_from_var(receiver)
            var_decl.move_prohibited = self._type_moveable(var_decl)
        return ast.FunctionCall(func.name, args, receiver,
                                type_args=type_args, trait_func=trait_func)

    # Where

    def _gen_func_call_ref(self,
                           etype: tp.Type,
                           only_leaves=False,
                           subtype=False) -> ast.FunctionCall:
        """Generate a function call from a reference.

        This function searches for variables and receivers in current scope.

        Args:
            etype: the type that the function call should return.
            only_leaves: do not generate new leaves except from `expr`.
            subtype: The returned type could be a subtype of `etype`.
        """
        # Tuple of signature, name, receiver
        refs = []
        # Search for function references in current scope
        variables = self.context.get_vars(self.namespace).values()
        if self._inside_inner_function: # function references only in current scope for Rust
            variables = list(self.context.get_vars(namespace=self.namespace, only_current=True).values())
        for var in variables:
            if var.is_moved:
                continue
            var_type = var.get_type()
            if var_type.is_type_var() and var_type.bound is not None and var_type.bound.name == "Fn":
                func_type_constr = self.bt_factory.get_function_type(len(var_type.bound.type_args) - 1)
                func_type = func_type_constr.new(var_type.bound.type_args)
                var_type = func_type
            if not getattr(var_type, 'is_function_type', lambda: False)():
                continue
            ret_type = var_type.type_args[-1]
            if (subtype and ret_type.is_assignable(etype)) or ret_type == etype:
                refs.append((var_type, var.name, None))
        if not refs:
            # Detect receivers
            objs = self._get_matching_objects(etype, subtype, 'fields',
                                              signature=False, func_ref=True)
            refs = [(tp.substitute_type(
                        obj.attr_decl.get_type(), obj.receiver_inst),
                    obj.attr_decl.name,
                    obj.receiver_expr)
                    for obj in objs
                   ]

        if not refs:
            return None

        signature, name, receiver = ut.random.choice(refs)
        if isinstance(receiver, ast.Variable):
            var_decl = self._get_decl_from_var(receiver)
            var_decl.move_prohibited = self._type_moveable(var_decl)
        # Generate arguments
        args = []
        initial_depth = self.depth
        self.depth += 1
        prev_move_semantics = self.move_semantics
        self.move_semantics = True
        for param_type in signature.type_args[:-1]:
            gen_bottom = param_type.is_wildcard() or (
                param_type.is_parameterized() and param_type.has_wildcards())
            arg = self.generate_expr(param_type, only_leaves,
                                     gen_bottom=gen_bottom, sam_coercion=False)
            args.append(ast.CallArgument(arg))
        self.depth = initial_depth
        self.move_semantics = prev_move_semantics
        return ast.FunctionCall(name, args, receiver=receiver,
                                is_ref_call=True)

    def get_struct_that_impls(self, trait_type):
        variables = self.context.get_vars(self.namespace).values()
        for var in variables:
            var_name = var.name
            if var.is_moved or var.move_prohibited:
                continue
            if var_name not in self._impls.keys():
                continue
            for (impl, _, _) in self._impls[var_name]:
                if impl.trait == trait_type:
                    var.is_moved = var.move_prohibited = True
                    return ast.Variable(var_name)
        for struct_name, lst in self._impls.items():
            for (impl, _, _) in lst:
                if impl.trait == trait_type:
                    return self.gen_new(impl.struct)
        assert False, "There should be a struct with a matching impl"
  
    # pylint: disable=unused-argument
    def gen_new(self,
                etype: tp.Type,
                only_leaves=False,
                subtype=True,
                sam_coercion=False) -> ast.New:
        """Create a new object of a given type.
        Args:
            etype: the type for which we want to create an object
            only_leaves: do not generate new leaves except from `expr`.
            subtype: The type could be a subtype of `etype`.
            sam_coercion: Apply sam coercion if possible.
        """
        if getattr(etype, 'is_function_type', lambda: False)():
            return self._gen_func_ref_lambda(etype, only_leaves=only_leaves)
        
        s_decl = None
        for struct in self.context.get_structs(self.namespace).values():
            if struct.name == etype.name:
                s_decl = struct
        news = {
            self.bt_factory.get_any_type(): ast.New(
                self.bt_factory.get_any_type(), args=[]),
            self.bt_factory.get_void_type(): ast.New(
                self.bt_factory.get_void_type(), args=[])
        }
        con = news.get(etype)
        if con is not None:
            return con
        
        if s_decl is None or etype.name in self._blacklisted_structs:
            t = etype
            if etype.is_type_var() and (
                    etype.name not in self._get_type_variable_names()):
                t = None
            return ast.BottomConstant(t)
        
        if etype.is_type_constructor():
            etype, _ = self.instantiate_type_constructor(etype, self.get_types())
        if s_decl.is_parameterized() and (
              s_decl.get_type().name != etype.name):
            etype, _ = self.instantiate_type_constructor(s_decl.get_type(), self.get_types())
        # If the matching struct is a parameterized one, we need to create
        # a map mapping the struct's type parameters with the corresponding
        # type arguments as given by the `etype` variable.
        type_param_map = (
            {} if not s_decl.is_parameterized()
            else {t_p: etype.type_args[i]
                  for i, t_p in enumerate(s_decl.type_parameters)}
        )
        initial_depth = self.depth
        self.depth += 1
        args = []
        prev_move_semantics = self.move_semantics
        self.move_semantics = True
        for field in s_decl.fields:
            expr_type = tp.substitute_type(field.get_type(), type_param_map)
            gen_bottom = expr_type.name == etype.name or (self.depth > (
                cfg.limits.max_depth * 2) and not expr_type.is_primitive())
            args.append(self.generate_expr(expr_type, only_leaves,
                                           subtype=False,
                                           gen_bottom=gen_bottom,
                                           sam_coercion=True))
        self.depth = initial_depth
        new_type = s_decl.get_type()
        if s_decl.is_parameterized():
            new_type = new_type.new(etype.type_args)
        field_names = [f.name for f in s_decl.fields]
        self.move_semantics = prev_move_semantics
        return ast.StructInstantiation(s_decl.name, etype, field_names, args)

    # Where

    def _gen_func_ref_lambda(self, etype:tp.Type, only_leaves=False):
        """Generate a function reference or a lambda for a given signature.

        Args:
            etype: signature

        Returns:
            ast.Lambda or ast.FunctionReference
        """
        ret_type, params = self._gen_ret_and_paramas_from_sig(etype)
        return self.gen_lambda(etype=ret_type, params=params,
                               only_leaves=only_leaves)

    # Where

    def _gen_func_ref(self, etype: tp.Type,
                      only_leaves=False) -> List[ast.FunctionReference]:
        """Generate a function reference.

        1. Functions in current scope and global scope, or methods that have
            a receiver in current scope.
        2. Create receiver for a function reference.
        3. Create a new function.

        Args:
            etype: signature for function reference
        """
        # Get function references from functions in the current scope or
        # methods that have a receiver in the current scope.
        refs = []
        funcs = self._get_matching_function_declarations(
            etype, False, signature=True)
        for func in funcs:
            if func.attr_decl.name == self.namespace[-1]:
                continue
            if func.receiver_expr is not None: #ignoring functions with receivers
                continue
            refs.append(ast.FunctionReference(
                func.attr_decl.name, func.receiver_expr, etype))
        if refs:
            return ut.random.choice(refs)
        ref = None
        type_fun = None
        # Generate a matching function.
        if not type_fun:
            type_fun = self._gen_matching_func(
                etype, not_void=True, signature=True, not_struct_func=True)
        if type_fun:
            receiver = (
                None if type_fun.receiver_t is None
                else self.generate_expr(type_fun.receiver_t,
                                        only_leaves=only_leaves)
            )
            ref = ast.FunctionReference(
                type_fun.attr_decl.name, receiver, etype)
        return ref

    ### Standard API of Generator ###

    def get_generators(self,
                       expr_type: tp.Type,
                       only_leaves: bool,
                       subtype: bool,
                       exclude_var: bool,
                       sam_coercion=False) -> List[Callable]:
        """Get candidate generators for the given type.

        Args:
            expr_type: targeted type.
            only_leaves: do not generate new leaves except from `expr`.
            subtype: The type of the generated expression could be a subtype
                of `expr_type`.
            exclude_var: if this option is false, then it could assign the
                generated expression into a variable, and return that
                variable reference.
            sam_coercion: Enable sam coercion.

        Returns:
            A list of generator functions
        """
        def gen_variable(etype):
            return self.gen_variable(etype, only_leaves, subtype)

        def gen_fun_call(etype):
            return self.gen_func_call(etype, only_leaves=only_leaves,
                                      subtype=subtype)

        # Do not generate new nodes in context.
        leaf_candidates = [
            lambda x: self.gen_new(x, only_leaves, subtype,
                                   sam_coercion=sam_coercion),
        ]
        constant_candidates = {
            self.bt_factory.get_number_type().name: gens.gen_integer_constant,
            self.bt_factory.get_integer_type().name: gens.gen_integer_constant,
            self.bt_factory.get_big_integer_type().name: gens.gen_integer_constant,
            self.bt_factory.get_byte_type().name: gens.gen_integer_constant,
            self.bt_factory.get_short_type().name: gens.gen_integer_constant,
            self.bt_factory.get_long_type().name: gens.gen_integer_constant,
            self.bt_factory.get_float_type().name: gens.gen_real_constant,
            self.bt_factory.get_double_type().name: gens.gen_real_constant,
            self.bt_factory.get_big_decimal_type().name: gens.gen_real_constant,
            self.bt_factory.get_char_type().name: gens.gen_char_constant,
            self.bt_factory.get_string_type().name: gens.gen_string_constant,
            self.bt_factory.get_boolean_type().name: gens.gen_bool_constant,
            self.bt_factory.get_array_type().name: (
                lambda x: self.gen_array_expr(x, only_leaves, subtype=subtype)
            ),
        }
        binary_ops = {
            self.bt_factory.get_boolean_type(): [
                lambda x: self.gen_logical_expr(x, only_leaves),
                lambda x: self.gen_equality_expr(only_leaves),
                lambda x: self.gen_comparison_expr(only_leaves)
            ],
        }
        other_candidates = [
            lambda x: self.gen_field_access(x, only_leaves, subtype),
            lambda x: self.gen_conditional(x, only_leaves=only_leaves,
                                           subtype=subtype),
            gen_fun_call,
            gen_variable
        ]

        if len(self.namespace) > 3 and self.namespace[1][0].isupper() and self.namespace[-2][0].islower():
            other_candidates.remove(gen_fun_call) #removing function calls for inner trait functions for Rust
            if expr_type == self.bt_factory.get_void_type():
                return [lambda x: ast.BottomConstant(x)]

        if expr_type == self.bt_factory.get_void_type():
            return [gen_fun_call,
                    lambda x: self.gen_assignment(x, only_leaves)]

        if isinstance(expr_type, tp.TypeParameter):
            leaf_candidates = []
        if self.depth >= cfg.limits.max_depth or only_leaves:
            gen_con = constant_candidates.get(expr_type.name)
            if gen_con is not None:
                return [gen_con]
            gen_var = (
                self._vars_in_context.get(
                    self.namespace, 0) < cfg.limits.max_var_decls and not
                only_leaves and not exclude_var)
            if gen_var:
                # Decide if we can generate a variable.
                # If the maximum numbers of variables in a specific context
                # has been reached, or we have previously declared a variable
                # of a specific type, then we should avoid variable creation.
                leaf_candidates.append(gen_variable)
            elif isinstance(expr_type, tp.TypeParameter):
                return [lambda x: self.generate_expr(x, gen_bottom=True)]
            return leaf_candidates
        con_candidate = constant_candidates.get(expr_type.name)
        if con_candidate is not None:
            candidates = [con_candidate] + binary_ops.get(expr_type, [])
            if not exclude_var:
                candidates.append(gen_variable)
        else:
            candidates = leaf_candidates
        return other_candidates + candidates

    def get_types(self,
                  ret_types=True,
                  exclude_arrays=False,
                  exclude_covariants=False,
                  exclude_contravariants=False,
                  exclude_type_vars=False,
                  exclude_function_types=False,
                  exclude_usr_types=False) -> List[tp.Type]:
        """Get all available types.

        Including user-defined types, built-ins, and function types.
        Note that this may include Type Constructors.

        Args:
            ret_types: use non-nothing built-in types (use this option if you
                want to generate a return type).
            exclude_arrays: exclude array types.
            exclude_covariants: exclude covariant type parameters.
            exclude_contravariants: exclude contravariant type parameters.
            exclude_type_vars: exclude type variables.
            exclude_function_types: exclude function types.

        Returns:
            A list of available types.
        """

        usr_types = [s.get_type() 
            for s in self.context.get_structs(self.namespace).values()
        ] if not exclude_usr_types else []
        if self.namespace[-1][0].isupper():
            usr_types = [t for t in usr_types if t.name != self.namespace[-1]] #struct type cannot be recursive

        type_params = []
        if not exclude_type_vars:
            t_params = self.context.get_types(namespace=self.namespace, only_current=True) \
                if self._inside_inner_function else self.context.get_types(self.namespace)
            for t_param in t_params.values():
                type_params.append(t_param)

        if type_params and ut.random.bool():
            return type_params

        builtins = list(self.ret_builtin_types
                        if ret_types
                        else self.builtin_types)
        
        if exclude_arrays:
            builtins = [
                t for t in builtins
                if t.name != self.bt_factory.get_array_type().name
            ]
        if exclude_function_types:
            return usr_types + builtins
        return usr_types + builtins + self.function_types

    def select_type(self,
                    ret_types=True,
                    exclude_arrays=False,
                    exclude_covariants=False,
                    exclude_contravariants=False,
                    exclude_type_vars=False,
                    exclude_function_types=False,
                    exclude_usr_types=False) -> tp.Type:
        """Select a type from the all available types.

        It will always instantiating type constructors to parameterized types.

        Args:
            ret_types: use non-nothing built-in types (use this option if you
                want to generate a return type).
            exclude_arrays: exclude array types.
            exclude_covariants: exclude covariant type parameters.
            exclude_contravariants: exclude contravariant type parameters.
            exclude_type_vars: exclude type variables.
            exclude_function_types: exclude function types.

        Returns:
            Returns a type.
        """
        types = self.get_types(ret_types=ret_types,
                               exclude_arrays=exclude_arrays,
                               exclude_covariants=exclude_covariants,
                               exclude_contravariants=exclude_contravariants,
                               exclude_type_vars=exclude_type_vars,
                               exclude_function_types=exclude_function_types,
                               exclude_usr_types=exclude_usr_types)
        stype = ut.random.choice(types)
        if stype.is_type_constructor():
            exclude_type_vars = stype.name == self.bt_factory.get_array_type().name
            stype, _ = self.instantiate_type_constructor(
                stype, self.get_types(exclude_arrays=True,
                                      exclude_covariants=True,
                                      exclude_contravariants=True,
                                      exclude_type_vars=exclude_type_vars,
                                      exclude_function_types=exclude_function_types,
                                      exclude_usr_types=exclude_usr_types))
            msg = "Instantiating type constructor {}".format(stype)
            log(self.logger, msg)
        return stype

    def gen_type_params(self,
                        count: int=None,
                        with_variance=False,
                        blacklist: List[str]=None,
                        for_function=False,
                        for_impl=False,
                        add_to_context=True) -> List[tp.TypeParameter]:
        """Generate a list containing type parameters

        Args:
            count: number of type parameters, if none it randomly select the
                number of type parameters.
            with_variance: enable variance
            blacklist: a list of type parameter names
            for_function: create type parameters for parameterized functions
            add_to_context: add type params to the context (False if called for impl block)
        """
        if not count and ut.random.bool():
            return []
        type_params = []
        type_param_names = blacklist or []
        limit = (
            # In case etype is Function3<T1, T2, T3, F_N>
            4
            if count == 4 and cfg.limits.max_type_params < 4
            else cfg.limits.max_type_params
        )
        iters = count or ut.random.integer(1, limit)
        for _ in range(iters):
            name = ut.random.caps(blacklist=type_param_names)
            type_param_names.append(name)
            if for_function:
                name = "F_" + name
            if for_impl:
                name = "I_" + name
            bound = None

            if ut.random.bool(cfg.prob.bounded_type_parameters):
                bound = self.choose_trait_bound()
            type_param = tp.TypeParameter(name, bound=bound)
            # Add type parameter to context.
            if add_to_context:
                self.context.add_type(self.namespace, type_param.name, type_param)
            type_params.append(type_param)
        return type_params

    def choose_trait_bound(self):
        """ Choose trait bound 
            Only traits that are implemented for at least one struct are considered 
        """
        candidates = []

        #Adding Fn trait bound
        nr_func_params = ut.random.choice(self.function_types).nr_type_parameters
        fn_trait_class = ut.random.choice(self.bt_factory.get_fn_trait_classes()) #Choose available function trait (Fn, ...)
        fn_trait = tu.instantiate_type_constructor(fn_trait_class(nr_func_params), self.get_types(exclude_type_vars=True))[0]
        candidates.append(fn_trait)

        #Adding other trait bounds
        for (_, lst) in self._impls.items():
            for (impl, _, _) in lst:
                bound = impl.trait
                if impl.type_parameters:
                    #Randomly instantiate impl's type parameters with some concrete types
                    type_map = {t_param: self.select_type() 
                                 if t_param.bound is None 
                                 else self.concretize_type(t_param.bound, {}, defaultdict(int))
                                 for t_param in impl.type_parameters
                               }
                    bound = tp.substitute_type(bound, type_map)
                candidates.append(bound)
        if not candidates:
            return None
        choice = ut.random.choice(candidates)
        return choice


    def _get_struct(self,
                    etype: tp.Type
                    ) -> Tuple[ast.StructDeclaration, tu.TypeVarMap]:
        """Find a struct declaration for the given type.
        """
        struct_decls = self.context.get_structs(self.namespace).values()
        for s in struct_decls:
            s_type = s.get_type()
            t_con = getattr(etype, 't_constructor', None)
            if s_type.name == etype.name or s_type == t_con:
                if s.is_parameterized():
                    type_var_map = {
                        t_param: etype.type_args[i]
                        for i, t_param in enumerate(s.type_parameters)
                    }
                else:
                    type_var_map = {}
                return s, type_var_map
        return None

    def _get_var_type_to_search(self, var_type: tp.Type) -> tp.TypeParameter:
        """Get the type that we want to search for.
        Args:
            var_type: The type of the variable.

        Returns:
            var_type or None
        """
        if tu.is_builtin(var_type, self.bt_factory) or var_type.is_type_var():
            return None
        return var_type

    def _get_vars_of_function_types(self, etype: tp.Type):
        """Get a variable or a field access whose type is a function type.

        Args:
            etype: function signature

        Returns:
            ast.Variable or ast.FieldAccess
        """
        refs = []
        # Get variables without receivers
        variables = list(self.context.get_vars(self.namespace).values())
        variables += list(self.context.get_vars(
            ('global',), only_current=True).values())
        for var_decl in variables:
            var_type = var_decl.get_type()
            var = ast.Variable(var_decl.name)
            if var_type == etype:
                refs.append(var)

        # field accesses
        objs = self._get_matching_objects(
                etype, False, 'fields', func_ref=True, signature=True)
        for obj in objs:
            refs.append(ast.FieldAccess(obj.receiver_expr, obj.attr_decl.name))
        return refs


    def _gen_func_body(self, ret_type: tp.Type):
        """Generate the body of a function or a lambda.

        Args:
            ret_type: Return type of the function

        Returns:
            ast.Block or ast.Expr
        """
        prev_move_semantics = self.move_semantics
        self.move_semantics = True
        log(self.logger, "Generating function body with return type: {}".format(ret_type))
        expr_type = (
            self.select_type(ret_types=False)
            if ret_type == self.bt_factory.get_void_type()
            else ret_type
        )
        expr = self.generate_expr(expr_type)
        self.move_semantics = prev_move_semantics
        decls = list(self.context.get_declarations(
            self.namespace, True).values())
        var_decls = [d for d in decls
                     if not isinstance(d, ast.ParameterDeclaration)]
        if (not var_decls and ret_type != self.bt_factory.get_void_type()):
            body = ast.Block([expr])
        else:
            exprs, decls = self._gen_side_effects()
            body = ast.Block(decls + exprs + [expr])
        return body


    def _gen_ret_and_paramas_from_sig(self, etype) -> \
            Tuple[tp.Type, ast.ParameterDeclaration]:
        """Generate parameters from signature and return them along with return
        type.

        Args:
            etype: signature type
            inside_lambda: true if we want to generate parameters for a lambda
        """
        params = [self.gen_param_decl(et) for et in etype.type_args[:-1]]
        ret_type = etype.type_args[-1]
        return ret_type, params

    # Matching functions

    def _get_matching_objects(self,
                              etype: tp.Type,
                              subtype: bool,
                              attr_name: str,
                              func_ref: bool = False,
                              signature: bool = False
                              ) -> List[gu.AttrReceiverInfo]:
        """Get objects that have an attribute of attr_name that is/return etype.

        Args:
            etype: the targeted type that we are searching. Functions should
                return that type.
            subtype: The type of matching attribute could be a subtype of
                `etype`.
            attr_name: 'fields' or 'functions'
            func_ref: look for function reference variables
            signature: etype is a signature.

        Returns:
            AttrReceiverInfo
        """
        decls = []
        variables = self.context.get_vars(self.namespace, only_current=self._inside_inner_function).values()
        for var in variables:
            if var.is_moved or (self.move_semantics and var.move_prohibited):
                continue
            
            #Adding function calls on variables with trait type bounds
            if var.get_type().is_type_var() and attr_name == 'functions':
                trait_bound_type = var.get_type().bound
                if not trait_bound_type or trait_bound_type.name not in self.context.get_traits(self.namespace).keys():
                    continue
                trait_decl = self.context.get_traits(self.namespace)[trait_bound_type.name]
                type_map = {}
                if trait_bound_type.is_parameterized():
                    type_map = dict(zip(trait_bound_type.t_constructor.type_parameters, trait_bound_type.type_args))
                for f in trait_decl.function_signatures + trait_decl.default_impls:
                    updated_f = self._update_func_decl(f, type_map)
                    if updated_f.get_type() == etype:
                        decls.append(gu.AttrReceiverInfo(
                            ast.Variable(var.name), {},
                            updated_f, {})
                        )
            var_type = self._get_var_type_to_search(var.get_type())
            if not var_type:
                continue
            if isinstance(getattr(var_type, 't_constructor', None),
                          self.function_type):
                continue
            s, type_var_map = self._get_struct(var_type)
            for attr in self._get_struct_attributes(s, attr_name, type_var_map):
                attr_type = tp.substitute_type(attr.get_type(), type_var_map)
                if attr_type == self.bt_factory.get_void_type():
                    continue
                if func_ref:
                    if not getattr(attr_type, 'is_function_type', lambda: False)():
                        continue

                if not self._is_sigtype_compatible(
                        attr, etype, type_var_map,
                        False,#signature and not func_ref
                        subtype,
                        lambda x, y: (
                            tp.substitute_type(
                                x.get_type(), y).type_args[-1]
                            if not signature and func_ref
                            else tp.substitute_type(x.get_type(), y)
                        )):
                    continue
                if getattr(attr, 'type_parameters', None):
                    decls.append(gu.AttrReceiverInfo(
                        ast.Variable(var.name), type_var_map,
                        attr, func_type_var_map
                    ))
                else:
                    decls.append(gu.AttrReceiverInfo(
                        ast.Variable(var.name), type_var_map, 
                        attr, None
                    ))
        return decls

    def _get_struct_attributes(self, struct_decl, attr_name, type_var_map=None):
        """Get attributes of a struct.
            If we want to get functions implemented for that struct,
            we need to pass type_var_map (to instantiate the function types correctly).
        Args:
            struct_decl: struct declaration
            attr_name: 'fields' or 'functions'
            type_var_map: mapping for parameterized structs (relevant for finding functions implemented for structs)
        """
        if attr_name == 'fields':
            return struct_decl.fields
        assert type_var_map is not None
        if not struct_decl.name in self._impls.keys():
            return []
        funcs = []
        for (impl, s_map, t_map) in self._impls[struct_decl.name]:
            is_compatible, updated_map = self._is_impl_compatible(s_map, type_var_map)
            if not is_compatible:
                continue
            for f in impl.avail_funcs:
                updated_f = self._update_func_decl(f, updated_map)
                funcs.append(updated_f)
        return funcs

    def _is_impl_compatible(self, impl_map, type_var_map):
        updated_map = {}
        for key in impl_map.keys():
            curr_inst = impl_map[key]
            if not curr_inst.has_type_variables():
                if curr_inst != type_var_map[key]:
                    return False, {}
            else:
                type_map = self.unify_types(type_var_map[key], curr_inst)
                if not type_map:
                    return False, {}
                for (t_param, t) in type_map.items():
                    if t_param in updated_map.keys() and updated_map[t_param] != t:
                        return False, {}
                    updated_map[t_param] = t
         
        return True, updated_map
                

    def _get_matching_function_declarations(self,
                                            etype: tp.Type,
                                            subtype: bool,
                                            signature=False
                                            ) -> List[gu.AttrReceiverInfo]:
        """Get all available function declarations.

        This function searches functions in the current scope that return
        `etype`, and then it also searches for receivers whose struct has a
        function that return `etype` (a function with a specific signature
        type).

        Args:
            etype: the return type for the function to find
                return that type.
            subtype: The return type of the function could be a subtype of
                `etype`.
            signature: etype is a signature.
        """
        functions = []
        is_nested_function = (
            self.namespace != ast.GLOBAL_NAMESPACE and
            self.namespace[-2].islower() and
            self.namespace[-2] != 'global'
        )
        msg = ("Searching for function declarations that match type {};"
               " checking signature {}")
        log(self.logger, msg.format(etype, signature))
        for func in self.context.get_funcs(self.namespace).values():
            # The receiver object for this kind of functions is None.
            if func.get_type() == self.bt_factory.get_void_type():
                continue

            if is_nested_function and func.name in self.namespace:
                # Here, we disallow recursive calls because it may lead to
                # recursive call on lambda expressions.
                continue
            if is_nested_function and signature:
                # Here we disallow nested functions to be used as function
                # references
                continue

            type_var_map = {}
            if func.is_parameterized():
                func_type_var_map = tu.unify_types(etype, func.get_type(),
                                                   self.bt_factory)
                if not func_type_var_map:
                    continue
                func_type_var_map = self.instantiate_parameterized_function(func.type_parameters, self.get_types(), func_type_var_map)
                type_var_map.update(func_type_var_map)

            if not self._is_sigtype_compatible(func, etype, type_var_map,
                                               signature, subtype):
                continue
            functions.append(gu.AttrReceiverInfo(None, {}, func, type_var_map))
        return functions + self._get_matching_objects(etype, subtype,
                                                      'functions',
                                                      signature=signature)

    def _get_struct_with_matching_function(self,
                                           etype: tp.Type) -> gu.AttrAccessInfo:
        """ Get a struct that implements a function that returns etype """
        structs = []
        for struct_name, lst in self._impls.items():
            for (impl, s_map, t_map) in lst:
                
                #instantiate impl type params randomly, concretizing bounded type params
                impl_type_map = {t_param: self.select_type() 
                                 if t_param.bound is None 
                                 else self.concretize_type(t_param.bound, {}, defaultdict(int))
                                 for t_param in impl.type_parameters
                                }
                for f in impl.avail_funcs:
                    if f.get_type() == self.bt_factory.get_void_type():
                        continue
                    if not f.get_type().has_type_variables() or etype == self.bt_factory.get_void_type():
                        if f.get_type() != etype: #types do not match
                            continue
                    else:
                        type_map = self.unify_types(etype, f.get_type())
                        if not type_map:
                            continue
                        impl_type_map.update(type_map)
                    #At this point, the function has the matching/compatible return type
                    updated_s_map = {}
                    for key in s_map.keys():
                        curr_inst = deepcopy(s_map[key])
                        updated_s_map[key] = tp.substitute_type(curr_inst, impl_type_map)
                    updated_f = self._update_func_decl(f, impl_type_map)
                    updated_s_type = tp.substitute_type(impl.struct, impl_type_map)
                    structs.append(gu.AttrAccessInfo(updated_s_type, updated_s_map, updated_f, None))
        if not structs:
            return None
        return ut.random.choice(structs)

    def _gen_matching_func(self,
                           etype: tp.Type,
                           not_void=False,
                           signature=False,
                           not_struct_func=False,
                           ) -> gu.AttrAccessInfo:
        """ Generate a function or a impl containing a method whose return
        type is 'etype'.

        Args:
            etype: the targeted return type.
            not_void: do not create functions that return void.
            signature: etype is a signature.
        """
        # Randomly choose to generate a function or a trait function.
        gen_call_on_struct = (
            ut.random.bool() and
            not not_struct_func and
            not etype.is_type_var() and
            not etype.is_parameterized() and
            not etype == self.bt_factory.get_void_type()
        )
        if not gen_call_on_struct:
            initial_namespace = self.namespace
            # If the given type 'etype' is a type parameter, then the
            # function we want to generate should be in the current namespace,
            # so that the type parameter is accessible.
            self.namespace = (
                self.namespace
                if ut.random.bool() or etype.has_type_variables()
                else ast.GLOBAL_NAMESPACE
            )
            # Generate a function
            params = None
            if signature:
                etype, params = self._gen_ret_and_paramas_from_sig(etype)
            func = self.gen_func_decl(etype, params=params, not_void=not_void)
            self.namespace = initial_namespace
            func_type_var_map = {}
            if func.is_parameterized():
                func_type_var_map = self.instantiate_parameterized_function(func.type_parameters, self.get_types(), {})
                func_ret_type = func.get_type()
                mapping = tu.unify_types(func_ret_type, etype, self.bt_factory)
                func_type_var_map.update(mapping)
                
            msg = "Generating a method {} of type {}; TypeVarMap {}".format(
                func.name, etype, func_type_var_map)
            log(self.logger, msg)

            return gu.AttrAccessInfo(None, {}, func, func_type_var_map)
        return self.gen_matching_impl(etype)


    def _is_sigtype_compatible(self, attr, etype, type_var_map,
                               check_signature, subtype,
                               get_attr_type=lambda x, y: tp.substitute_type(
                                   x.get_type(), y)):
        
        attr_type = get_attr_type(attr, type_var_map)
        if not check_signature:
            if subtype:
                return attr_type.is_assignable(etype)
            return attr_type == etype
        param_types = [
            tp.substitute_type(p.get_type(), type_var_map)
            for p in attr.params
        ]
        sig = tp.ParameterizedType(
            self.bt_factory.get_function_type(len(attr.params)),
            param_types + [attr_type])
        return etype == sig

    def _is_signature_compatible(self, attr, etype, check_signature,
                                 subtype):
        """
        Checks if the signature of attr is compatible with etype.
        """
        type_var_map = {}
        attr_type = attr.get_type()
        if check_signature:
            signature_types = [
                p.get_type() for p in attr.params
            ]
            signature_types.append(attr_type)
            # The signature of the function `attr` does not match with `etype`.
            # Namely, attr does not contain the same number of parameters
            # as `etype`.
            if len(signature_types) != len(etype.type_args):
                return False, None

            for i, st in enumerate(signature_types):
                if not st.has_type_variables():
                    continue
                # Unify its component of attr with the corresponding type
                # argument of etype.
                new_tvm = tu.unify_types(
                    etype.type_args[i], st,
                    self.bt_factory
                )
                if not new_tvm:
                    return False, None
                for k, v in new_tvm.items():
                    assigned_t = type_var_map.get(k, v)
                    # The instantiation of type variable k clashes with
                    # a previous instantiation of this type variable.
                    if assigned_t != v:
                        return False, None
                type_var_map.update(new_tvm)
        else:
            # if the type of the attribute has type variables,
            # then we have to unify it with the expected type so that
            # we can instantiate the corresponding type constructor
            # accordingly
            if attr_type.has_type_variables():
                type_var_map = tu.unify_types(etype, attr_type,
                                              self.bt_factory)
        is_comb = self._is_sigtype_compatible(attr, etype, type_var_map,
                                              check_signature, subtype)
        return is_comb, type_var_map

    def _create_type_params_from_etype(self, etype: tp.Type):
        type_params = []
        retrieved_type_params = self._retrieve_type_params(etype)
        for t_param in retrieved_type_params:
            if t_param not in type_params:
                type_params.append(t_param)
        mapping = {tv : tv for tv in type_params}
        return type_params, mapping, False

    def _retrieve_type_params(self, t):
        if t.is_type_var():
            if t.bound is None or not t.bound.has_type_variables():
                return [t]
            return self._retrieve_type_params(t.bound) + [t]
        if t.is_parameterized():
            res = []
            for t_p in t.type_args:
                res.extend(self._retrieve_type_params(t_p))
            return res
        return []



    def gen_struct_decl(self,
                        struct_name: str=None,
                        field_type: tp.Type=None,
                        init_type_params: List[tp.TypeParameter]=None,
                        ) -> ast.StructDeclaration:
        """Generate a struct declaration.
        Args:
        field_type: At least one field will have this type.
        type_params: List with type parameters.

        Returns: A struct declaration node.
        """
        struct_name = struct_name or gu.gen_identifier('capitalize')
        initial_namespace = self.namespace
        self.namespace += (struct_name,)
        initial_depth = self.depth
        self.depth += 1
        init_type_params = init_type_params or []
        struct = ast.StructDeclaration(
            name=struct_name,
            fields=[],
            functions=[],
            impl_traits=[],
            type_parameters=[],
        )
        self._add_node_to_parent(ast.GLOBAL_NAMESPACE, struct)
        self._blacklisted_structs.add(struct_name)
        type_params = self.gen_type_params() if init_type_params is None else init_type_params
        fields = self.gen_struct_fields(field_type, init_type_params) #force struct to have fields using init_type_params
        type_params = self._remove_unused_type_params(type_params, [f.get_type() for f in fields])

        struct.fields = fields
        struct.type_parameters = type_params
        self._blacklisted_structs.remove(struct_name)
        self.namespace = initial_namespace
        self.depth = initial_depth
        return struct

    def _get_type_vars_from_type(self, t : tp.Type):
        if t.is_type_var():
            return [t]
        res = []
        if t.is_parameterized():
            for t_p in t.type_args:
                res.extend(self._get_type_vars_from_type(t_p))
        return res


    def gen_struct_fields(self, field_type: tp.Type=None, type_params: List[tp.TypeParameter]=None):
        max_fields = cfg.limits.cls.max_fields - 1 if field_type else cfg.limits.cls.max_fields
        fields = []
        if field_type:
            fields.append(self.gen_field_decl(field_type, True))
        if type_params:
            for t_param in type_params:
                fields.append(self.gen_field_decl(field_type, True))
        for _ in range(ut.random.integer(0, max_fields)):
            fields.append(self.gen_field_decl(None, True))
        return fields
    
    def gen_trait_decl(self, 
                       fret_type: tp.Type=None,
                       not_void: bool=False,
                       trait_name: str=None,
                       signature: tp.ParameterizedType=None,
                       type_params=None
                       ) -> ast.TraitDeclaration:
        """Generate a trait declaration."""
        self._inside_trait_decl = True
        trait_name = trait_name or gu.gen_identifier('capitalize')
        initial_namespace = self.namespace
        self.namespace += (trait_name,)
        initial_depth = self.depth
        type_params = type_params or self.gen_type_params()
        self.depth += 1
        trait = ast.TraitDeclaration(
            name=trait_name,
            function_signatures=[],
            default_impls=[],
            supertraits=[],
            type_parameters=type_params,
        )
        self._add_node_to_parent(ast.GLOBAL_NAMESPACE, trait)
        self._blacklisted_traits.add(trait_name)
        self.gen_trait_functions(fret_type=fret_type, not_void=not_void, signature=signature)
        self._blacklisted_traits.remove(trait_name)
        self._inside_trait_decl = False
        self.namespace = initial_namespace
        return trait


    def gen_trait_functions(self,
                            fret_type=None,
                            not_void=False,
                            signature: tp.ParameterizedType=None) -> List[ast.FunctionDeclaration]:
        """ Generate trait signatures and default implementations.
        """
        funcs = []
        max_funcs = cfg.limits.cls.max_funcs - 1 if fret_type else cfg.limits.cls.max_funcs
        if fret_type:
            func = self.gen_func_decl(fret_type, not_void=not_void, is_signature=ut.random.bool(), trait_func=True)
            funcs.append(func)
        if signature:
            ret_type, params = self._gen_ret_and_paramas_from_sig(signature)
            func = self.gen_func_decl(ret_type, params=params, not_void=not_void, is_signature=ut.random.bool(), trait_func=True)
            func.trait_func = True
            funcs.append(func)
        for _ in range(ut.random.integer(0, max_funcs)):
            func = self.gen_func_decl(not_void=not_void, is_signature=ut.random.bool(), trait_func=True)
            funcs.append(func)
        return funcs


    def gen_matching_impl(self, fret_type: tp.Type) -> gu.AttrAccessInfo:
        impl = self.gen_impl(fret_type)
        return self._get_struct_with_matching_function(fret_type)
 
    def gen_impl(self,
                 fret_type: tp.Type=None,
                 struct_trait_pair: Tuple[tp.Type, tp.Type]=None,
                 ) -> ast.Impl:
        """Generate an impl block.
           Args:
                fret_type: if provided, at least one function will return this type.
                struct_trait_pair: if provided, impl is generated for this pair
                    Otherwise struct and trait are chosen randomly.
            Returns:
                An impl declaration node.
        """
        def _inst_type_constructor(obj):
            ttype, type_var_map = obj.get_type(), {}
            if obj.is_parameterized():
                ttype, type_var_map = self.instantiate_type_constructor(obj.get_type(), self.get_types() + impl_type_params)
            return ttype, type_var_map

        initial_depth = self.depth
        self.depth += 1
        initial_namespace = self.namespace
        self.namespace = ast.GLOBAL_NAMESPACE
        impl_type_params = []
        if struct_trait_pair:
            s_type, t_type = struct_trait_pair
            struct = self.context.get_structs(self.namespace)[s_type.name]
            trait = self.context.get_traits(self.namespace)[t_type.name]
            struct_map = {t_param: s_type.type_args[i] for i, t_param in enumerate(s_type.t_constructor.type_parameters)} \
                if s_type.is_parameterized() else {}
            trait_map = {t_param: t_type.type_args[i] for i, t_param in enumerate(t_type.t_constructor.type_parameters)} \
                if t_type.is_parameterized() else {}
            if s_type.has_type_variables():
                impl_type_params = self._get_type_vars_from_type(s_type)
        else:
            #Generate candidate type parameters
            #Type params used in final declaration of impl block will be a subset of these,
            #depending on instantiations of struct and trait
            impl_type_params = self.gen_type_params(add_to_context=False, count=cfg.limits.max_type_params, for_impl=True)
            #find or generate trait
            if fret_type is not None:
                trait = self._get_matching_trait(fret_type)
                if not trait:
                    trait = self._gen_matching_trait(fret_type, True)
            else:
                available_traits = [trait for trait in list(self.context.get_traits(self.namespace).values()) \
                    if trait.name not in self._blacklisted_traits]
                trait = self.gen_trait_decl() if not available_traits \
                    else ut.random.choice(available_traits)
            #find or generate struct
            structs_in_context = list(self.context.get_structs(self.namespace).values())
            struct = self.gen_struct_decl() if not structs_in_context \
                else ut.random.choice(structs_in_context)
            s_type, struct_map = _inst_type_constructor(struct)
            if not self._is_impl_allowed(struct, struct_map, trait):
                struct = self.gen_struct_decl() #if not allowed, just generate a fresh struct
                s_type, struct_map = _inst_type_constructor(struct)
            #Type parameters used in final declaration of impl block are the ones that are also used in struct type instantiation
            s_type_params = sum([self._get_type_vars_from_type(t_arg) for t_arg in s_type.type_args], []) \
                if s_type.is_parameterized() else []
            impl_type_params = [t_param for t_param in impl_type_params if t_param in s_type_params]
            t_type, trait_map = _inst_type_constructor(trait)
        impl_id = self._get_impl_id(str(t_type), str(s_type))
        self.namespace = ast.GLOBAL_NAMESPACE + (impl_id,)
        
        #Adding impl block info to _impls
        impl = ast.Impl(name=impl_id, 
                        struct=s_type, 
                        trait=t_type, 
                        functions=[], 
                        avail_funcs=[], 
                        type_parameters=impl_type_params)
        if not struct.name in self._impls.keys():
            self._impls[struct.name] = []
        self._impls[struct.name].append((impl, struct_map, trait_map))

        #Adding type parameters into context, so that they can be used in generated functions
        for t_param in impl_type_params:
            self.context.add_type(self.namespace, t_param.name, t_param)
        
        #Adding fields to _fields_vars so that they are accessible in impl block
        field_vars_list = []
        for field in struct.fields:
            field_type = tp.substitute_type(field.get_type(), struct_map)
            field_var_name ="self." + field.name
            #create a virtual variable declaration for each struct field
            var_decl = ast.VariableDeclaration(name=field_var_name, expr=None, var_type=field_type)
            var_decl.move_prohibited = self._type_moveable(var_decl)
            field_vars_list.append(var_decl)
        self._field_vars[impl_id] = field_vars_list
        
        for func_decl in trait.function_signatures + trait.default_impls:
            prev_ns = self.namespace
            self.namespace += (func_decl.name,)
            new_func = self._update_func_decl(func_decl, trait_map)
            new_func.body = self._gen_func_body(new_func.get_type())
            if func_decl in trait.default_impls:
                #decide randomly to override the default implementation
                if ut.random.bool():
                    impl.functions.append(new_func)
            else:
                impl.functions.append(new_func)
            impl.avail_funcs.append(new_func)
            self.namespace = prev_ns
        
        msg = ("Creating impl {}").format(impl_id)
        log(self.logger, msg)

        self._add_node_to_parent(ast.GLOBAL_NAMESPACE, impl)
        self.namespace = initial_namespace
        self.depth = initial_depth
        return impl


    def _update_func_decl(self, func_decl: ast.FunctionDeclaration, t_map: Dict):
        """Update the type of a function declaration with the given type map.
        """
        new_func = deepcopy(func_decl)
        new_func.inferred_type = new_func.ret_type = (t_map[func_decl.ret_type] if isinstance(func_decl.ret_type, tp.TypeParameter) 
        else tp.substitute_type(func_decl.ret_type, t_map))
        for i, param in enumerate(new_func.params):
            new_func.params[i].param_type = t_map[param.param_type] if isinstance(param.param_type, tp.TypeParameter) \
                else tp.substitute_type(param.param_type, t_map)
        return new_func

    def _get_impl_id(self, trait: str, struct: str):
        """ create a unique identifier for impl block """
        return "impl_" + trait + "_" + struct

    def _is_impl_allowed(self, struct_decl, struct_type_map, trait_decl):
        """ check if there are conflicting implementations 
            current check policy is strict: no two impl blocks with the same trait name
            and conflicting struct instantiations are allowed
        """
        if not struct_decl.name in self._impls.keys():
            return True
        for (impl, s_map, t_map) in self._impls[struct_decl.name]:
            if impl.trait.name == trait_decl.name:
                diff = False #we need different instantiation with concrete types for at least one type parameter
                for key in s_map.keys():
                    if not s_map[key].has_type_variables() and \
                    not struct_type_map[key].has_type_variables() and \
                    s_map[key] != struct_type_map[key]:
                        diff = True
                if not diff:
                    return False
        return True

    def _get_matching_trait(self, fret_type: tp.Type):
        trait_decls = self._get_matching_trait_decls(fret_type)
        if not trait_decls:
            return None
        t, type_var_map, func = ut.random.choice(trait_decls)
        return t

    def _get_matching_trait_decls(self, 
                                  fret_type: tp.Type) -> List[Tuple[ast.TraitDeclaration, tu.TypeVarMap, ast.FunctionDeclaration]]:
        """ Get traits that have functions that return fret_type.
        """
        trait_decls = []
        for t in self.context.get_traits(self.namespace).values():
            if t.name in self._blacklisted_traits:
                continue
            for func in t.function_signatures + t.default_impls:
                func_type = func.get_type()
                if func_type == self.bt_factory.get_void_type():
                    continue
                if fret_type.is_parameterized():
                    is_comp, type_var_map = self._is_signature_compatible(func, fret_type, True, False)
                    if is_comp:
                        trait_decls.append((t, type_var_map, func))
                else:
                    if func_type == fret_type:
                        trait_decls.append((t, {}, func))
        return trait_decls
        
    def _gen_matching_trait(self,
                            fret_type: tp.Type,
                            not_void=False):
        """Generate a trait that has a function that returns fret_type.
        """
        initial_namespace = self.namespace
        trait_name = gu.gen_identifier('capitalize')
        type_params = None
        if fret_type.has_type_variables():
            self.namespace = ast.GLOBAL_NAMESPACE + (trait_name,)
            type_params, type_var_map, can_wildcard = self._create_type_params_from_etype(fret_type)
            fret_type2 = tp.substitute_type(fret_type, type_var_map)
        else:
            type_var_map, fret_type2, can_wildcard = {}, fret_type, False
        self.namespace = ast.GLOBAL_NAMESPACE
        t = self.gen_trait_decl(fret_type=fret_type2, not_void=not_void, type_params=type_params, trait_name=trait_name)
        self.namespace = initial_namespace
        return t