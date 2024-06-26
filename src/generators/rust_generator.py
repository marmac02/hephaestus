#GENERATOR for Rust

"""
This file includes the program generator for Rust specific features.
"""
# pylint: disable=too-many-instance-attributes,too-many-arguments,dangerous-default-value
from src.generators.generator import Generator
import functools
from collections import defaultdict
from copy import deepcopy
from typing import Tuple, List, Callable, Dict

from src import utils as ut
from src.generators import generators as gens
from src.generators import utils as gu
from src.generators.config import cfg
from src.ir import ast, types as tp, type_utils as tu, kotlin_types as kt
from src.ir.context import Context
from src.ir.builtins import BuiltinFactory
from src.ir import BUILTIN_FACTORIES
from src.modules.logging import Logger, log

class RustGenerator(Generator):
    # TODO document
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
        self._new_from_class = None
        self.namespace = ('global',)
        self.enable_pecs = not language == 'kotlin'
        self.disable_variance_functions = True #disabled for now
        self._field_vars = {} #maps impl block ids to available field variables
        self.move_semantics = False #flag to handle move semantics in Rust
        self.instantiation_depth = 2 #depth of type instantiation during concretization

        #Map describing impl blocks. It maps struct_names to tuples
        self._impls = {}

        # This flag is used for Java lambdas where local variables references
        # must be final.
        self._inside_java_lambda = False

        #This flag is used for Rust inner functions that cannot capture outer variables
        self._inside_inner_function = False
        self.flag_fn = False
        self.flag_Fn = False
        self.flag_FnMut = False
        self.flag_FnOnce = False

        #This flag is used for Rust to handle function calls in inner functions inside trait declarations
        self._inside_trait_decl = False
        
        self.function_type = type(self.bt_factory.get_function_type())
        self.function_types = self.bt_factory.get_function_types(
            cfg.limits.max_functional_params)
        self.function_trait_types = self.bt_factory.get_function_trait_types(
            cfg.limits.max_functional_params)
        self.ret_builtin_types = self.bt_factory.get_non_nothing_types()
        self.builtin_types = self.ret_builtin_types + \
            [self.bt_factory.get_void_type()]
        # In some case we need to use two namespaces. One for having access
        # to variables from scope, and one for adding new declarations.
        # In most cases one namespace is enough, but in cases like
        # `generate_expr` in `_gen_func_params_with_default` we need both
        # namespaces. To use one namespace we must model scope better.
        # Almost always declaration_namespace is set to None to be ignored
        self.declaration_namespace = None
        self.int_stream = iter(range(1, 10000))
        self._in_super_call = False
        # We use this data structure to store blacklisted classes, i.e.,
        # classes that are incomplete (we do not have the information regarding
        # their fields and functions yet). So, we avoid instantiating these
        # classes or using them as supertypes, because we do not have the
        # complete information about them.
        self._blacklisted_classes: set = set()
        self._blacklisted_traits: set = set() #for Rust
        self._blacklisted_structs: set = set() #for Rust
        tu.flag_for_rust = True
        func_type = self.function_types[1]

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

        * Variable Declarations
        * Class Declarations
        * Function Declarations

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
        #candidates.extend(self.lang_obs.get_top_levl_decl())
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
        self._add_node_to_parent(self.namespace, main_func)
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
                if visited_counter[impl.name] > self.instantiation_depth:
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
                if impl.trait.name != trait_type.name or visited_counter[impl.name] > self.instantiation_depth:
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
        '''
        if t.name.lower().startswith("fn"):
            #type is a function trait type, it must be concretized
            func_constr = self.bt_factory.get_function_type(len(t.type_args) - 1)
            concrete_fun_type = func_constr.new(t.type_args)
            return concrete_fun_type
        '''
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
    # FunctionDeclaration, ParameterDeclaration, ClassDeclaration,
    # FieldDeclaration, and VariableDeclaration

    def get_variables(self, namespace):
        """ Get all available variable declarations in the namespace and outer namespaces.
            Variable can be used if it fulfills conditions:
                - Declared in the current namespace or outer namespaces.
                - Respect move rules of Rust: if a variable has been moved, or
                  will be moved after the current statement and it would be used
                  in a move-inducing setting, it cannot be used. The latter case
                  occurs due to non-sequential generation logic of Hepheastus.
                - If a statement is generated for a closure body, it must comply
                  with function types: fn, Fn, FnMut, and FnOnce.
                  fn: only variables from current scope (local for lambda) and globals
                  Fn, FnMut: can capture external variables, but cannot move them
                  FnOnce: can capture and move external variables out of their environment.
        """
        variables = []
        for var in self.context.get_vars(namespace).values():
            if var.is_moved or (self.move_semantics and var.move_prohibited):
                continue
            if self.flag_fn:
                if self.context.get_namespace(var) not in {namespace, ast.GLOBAL_NAMESPACE}:
                    continue
            if self.flag_Fn or self.flag_FnMut:
                if self.context.get_namespace(var) not in {namespace, ast.GLOBAL_NAMESPACE} and \
                   self.move_semantics and not var.get_type().is_primitive():
                    continue
            variables.append(var)
        return variables

    def _get_decl_from_var(self, var):
        var_decls = self.context.get_vars(self.namespace)
        if var.name in var_decls.keys():
            return var_decls[var.name]
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
        if len(ns) > 2 and ns[1][0].isupper():
            trait_name = ns[1]
            trait_decl = self.context.get_traits(self.namespace)[trait_name]
            return trait_decl.type_parameters
        return []

    def gen_func_decl(self,
                      etype:tp.Type=None,
                      not_void=False,
                      class_is_final=False,
                      func_name:str=None,
                      params:List[ast.ParameterDeclaration]=None,
                      abstract=False,
                      is_interface=False,
                      trait_func=False,
                      type_params:List[tp.TypeParameter]=None,
                      namespace=None) -> ast.FunctionDeclaration:
        """Generate a function declaration.

        This method is responsible for generating all types of function/methods,
        i.e. functions, class methods, nested functions. Furthermore, it also
        generates parameterized functions.

        Args:
            etype: expected return type.
            not_void: do not return void.
            class_is_final: function of a final class.
            func_name: function name.
            params: list of parameter declarations.
            abstract: function of an abstract class.
            is_interface: function of an interface.
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
        # Check if this function we want to generate is a class method, by
        # checking the name of the outer namespace. If we are in class then
        # the outer namespace begins with capital letter.
        class_method = self.namespace[-2][0].isupper()
        class_method = (False if len(self.namespace) < 2 else
                        self.namespace[-2][0].isupper())
        can_override = abstract or is_interface or (class_method and not
                                    class_is_final and ut.random.bool())
        trait_func = self.namespace[-2][0].isupper() #check if function is declared in a trait
        # Check if this function we want to generate is a nested functions.
        # To do so, we want to find if the function is directly inside the
        # namespace of another function.
        nested_function = (len(self.namespace) > 1 and
                           self.namespace[-2] != 'global' and
                           self.namespace[-2][0].islower())

        prev_inside_java_lamdba = self._inside_java_lambda
        self._inside_java_lambda = nested_function and self.language == "java"
        prev_flag_fn = self.flag_fn
        self.flag_fn = nested_function
        # Type parameters of functions cannot be variant.
        # Also note that at this point, we do not allow a conflict between
        # type variable names of class and type variable names of functions.
        # TODO consider being less conservative.
        if (not nested_function and not trait_func): #or (nested_function and self.language == 'rust'): #nested functions in Rust can be parameterized
            if type_params is not None:
                for t_p in type_params:
                    # We add the types to the context.
                    self.context.add_type(self.namespace, t_p.name, t_p)
            else:
                # Type parameters of parameterized functions can be neither
                # covariant nor contravariant.
                type_params = self.gen_type_params(
                    with_variance=False,
                    blacklist=self._get_type_variable_names(),
                    for_function=True
                ) if ut.random.bool(prob=cfg.prob.parameterized_functions) \
                  else []

        else:
            # Nested functions cannot be parameterized (
            # at least in Groovy, Java), because they are modeled as lambdas.
            type_params = []
        if params is not None:
            for p in params:
                log(self.logger, "Adding parameter {} to context in gen_func_decl".format(p.name))
                self._add_node_to_parent(self.namespace, p)
        else:
            params = self._gen_func_params()
        ret_type = self._get_func_ret_type(params, etype, not_void=not_void)
        if is_interface or (abstract and ut.random.bool()):
            body, inferred_type = None, None
        else:
            # If we are going to generate a non-abstract method, we generate
            # a temporary body as a placeholder.
            body = ast.BottomConstant(ret_type)
        type_params = self._remove_unused_type_params(type_params, [p.get_type() for p in params] + [ret_type])
        type_params = self._add_used_type_params(type_params, params, ret_type) #for Rust
        if trait_func:
            params = [ast.SelfParameter()] + params #adding &self parameter for Rust trait functions
        func = ast.FunctionDeclaration(
            func_name, params, ret_type, body,
            func_type=(ast.FunctionDeclaration.CLASS_METHOD
                       if class_method
                       else ast.FunctionDeclaration.FUNCTION),
            is_final=not can_override,
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
        self._inside_java_lambda = prev_inside_java_lamdba
        self.flag_fn = prev_flag_fn
        self.depth = initial_depth
        self.namespace = initial_namespace
        return func


    def gen_param_decl(self, etype=None, for_lambda=False) -> ast.ParameterDeclaration:
        """Generate a function Parameter Declaration.

        Args:
            etype: Parameter type.
        """
        name = gu.gen_identifier('lower')
        if etype and etype.is_wildcard():
            bound = etype.get_bound_rec()
            param_type = bound or self.select_type(exclude_covariants=True)
        else:
            param_type = etype or self.select_type(include_fn_traits=not for_lambda)
        param = ast.ParameterDeclaration(name, param_type)
        return param

    def gen_class_decl(self,
                       field_type: tp.Type=None,
                       fret_type: tp.Type=None,
                       not_void: bool=False,
                       type_params: List[tp.TypeParameter]=None,
                       class_name: str=None,
                       signature: tp.ParameterizedType=None
                       ) -> ast.ClassDeclaration:
        """Generate a class declaration.

        It generates all type of classes (regular, abstract, interface),
        and it can also generate parameterized classes.

        Args:
            field_type: At least one field will have this type.
            fret_type: At least one function will return this type.
            not_void: Do not generate functions that return void.
            type_params: List with type parameters.
            class_name: Class name.
            signature: Generate at least one function with the given signature.

        Returns:
            A class declaration node.
        """
        class_name = class_name or gu.gen_identifier('capitalize')
        initial_namespace = self.namespace
        self.namespace += (class_name,)
        initial_depth = self.depth
        self.depth += 1
        class_type = gu.select_class_type(field_type is not None)
        is_final = ut.random.bool() and class_type == \
            ast.ClassDeclaration.REGULAR
        type_params = type_params or self.gen_type_params(
            with_variance=self.language in ['kotlin', 'scala'])
        cls = ast.ClassDeclaration(
            class_name,
            class_type=class_type,
            superclasses=[],
            type_parameters=type_params,
            is_final=is_final,
            fields=[],
            functions=[]
        )
        self._add_node_to_parent(ast.GLOBAL_NAMESPACE, cls)
        self._blacklisted_classes.add(class_name)

        super_cls_info = self._select_superclass(
            class_type == ast.ClassDeclaration.INTERFACE)
        if super_cls_info:
            cls.superclasses = [super_cls_info.super_inst]
            cls.supertypes = [c.class_type for c in cls.superclasses]
        if not cls.is_interface():
            self.gen_class_fields(cls, super_cls_info, field_type)

        self.gen_class_functions(cls, super_cls_info,
                                 not_void, fret_type, signature)
        self._blacklisted_classes.remove(class_name)
        self.namespace = initial_namespace
        self.depth = initial_depth
        return cls

    # Where

    def _select_superclass(self, only_interfaces: bool) -> gu.SuperClassInfo:
        """
        Select a superclass for a class.

        Args:
            only_interfaces: select an interface.

        Returns:
            SuperClassInfo object which includes: the super class declaration,
                its TypeVarMap, and a SuperClassInstantiation for the selected
                class.
        """

        current_cls = self.namespace[-1]

        def is_cls_candidate(cls):
            # A class should not inherit from itself to avoid circular
            # dependency problems.
            if cls.name == current_cls:
                return False
            if cls.name in self._blacklisted_classes:
                return False
            return not cls.is_final and (cls.is_interface()
                                         if only_interfaces else True)

        class_decls = [
            c for c in self.context.get_classes(self.namespace).values()
            if is_cls_candidate(c)
        ]
        if not class_decls:
            return None
        class_decl = ut.random.choice(class_decls)
        if class_decl.is_parameterized():
            cls_type, type_var_map = tu.instantiate_type_constructor(
                class_decl.get_type(),
                self.get_types(exclude_covariants=True,
                               exclude_contravariants=True,
                               exclude_arrays=True),
                enable_pecs=self.enable_pecs,
                disable_variance_functions=self.disable_variance_functions,
                only_regular=True,
            )
        else:
            cls_type, type_var_map = class_decl.get_type(), {}
        con_args = None if class_decl.is_interface() else []
        prev_super_call = self._in_super_call
        self._in_super_call = True
        for f in class_decl.fields:
            field_type = tp.substitute_type(f.get_type(), type_var_map)
            con_args.append(self.generate_expr(field_type,
                                               only_leaves=True))
        self._in_super_call = prev_super_call
        return gu.SuperClassInfo(
            class_decl,
            type_var_map,
            ast.SuperClassInstantiation(cls_type, con_args)
        )

    # And

    def gen_class_fields(self,
                         curr_cls: ast.ClassDeclaration,
                         super_cls_info: gu.SuperClassInfo,
                         field_type: tp.Type=None
                        ) -> List[ast.FieldDeclaration]:
        """Generate fields for a class.

        It also adds the fields in the context.

        Args:
            curr_cls: Current class declaration.
            super_cls_info: SuperClassInstantiation for curr_cls
            field_type: At least one field will have this type.

        Returns:
            A list of field declarations
        """
        max_fields = cfg.limits.cls.max_fields - 1 if field_type \
            else cfg.limits.cls.max_fields
        fields = []
        if field_type:
            fields.append(self.gen_field_decl(field_type, curr_cls.is_final))
        if not super_cls_info:
            for _ in range(ut.random.integer(0, max_fields)):
                fields.append(
                    self.gen_field_decl(class_is_final=curr_cls.is_final))
        else:
            overridable_fields = super_cls_info.super_cls \
                .get_overridable_fields()
            k = ut.random.integer(0, min(max_fields, len(overridable_fields)))
            if overridable_fields:
                chosen_fields = ut.random.sample(overridable_fields, k=k)
                for f in chosen_fields:
                    field_type = tp.substitute_type(
                        f.get_type(), super_cls_info.type_var_map)
                    new_f = self.gen_field_decl(field_type, curr_cls.is_final,
                                                add_to_parent=False)
                    new_f.name = f.name
                    new_f.override = True
                    new_f.is_final = f.is_final
                    fields.append(new_f)
                    log(self.logger, "Adding field {} to context in gen_class_fields".format(new_f.name))
                    self._add_node_to_parent(self.namespace, new_f)
                max_fields = max_fields - len(chosen_fields)
            if max_fields < 0:
                return fields
            for _ in range(ut.random.integer(0, max_fields)):
                fields.append(
                    self.gen_field_decl(class_is_final=curr_cls.is_final))
        return fields

    # Where

    def _add_node_to_class(self, cls, node):
        if isinstance(node, ast.FunctionDeclaration):
            cls.functions.append(node)
            return

        if isinstance(node, ast.FieldDeclaration):
            cls.fields.append(node)
            return

        assert False, ('Trying to put a node in class other than a function',
                       ' and a field')

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
            ast.ClassDeclaration: self.context.add_class,
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
            if isinstance(parent, ast.ClassDeclaration):
                self._add_node_to_class(parent, node)
            if isinstance(parent, ast.StructDeclaration):
                self._add_node_to_struct(parent, node)
            if isinstance(parent, ast.TraitDeclaration):
                self._add_node_to_trait(parent, node)

        node_type[type(node)](parent_namespace, node.name, node)


    # And

    def gen_class_functions(self,
                            curr_cls, super_cls_info,
                            not_void=False,
                            fret_type=None,
                            signature: tp.ParameterizedType=None
                            ) -> List[ast.FunctionDeclaration]:
        """Generate methods for a class.

        If the method has a superclass, then it will try to implement any
        method that must be implemented (e.g., abstract methods in regular
        classes).

        Args:
            curr_cls: Current Class declaration
            super_cls_info: SuperClassInstantiation for curr_cls
            not_void: Do not create methods that return void.
            fret_type: At least one method will return this type.
            signature: Generate at least one function with the given signature.
        """

        funcs = []
        max_funcs = cfg.limits.cls.max_funcs - 1 if fret_type \
            else cfg.limits.cls.max_funcs
        max_funcs = max_funcs - 1 if signature else max_funcs
        abstract = not curr_cls.is_regular()
        if fret_type:
            funcs.append(
                self.gen_func_decl(fret_type, not_void=not_void,
                                   class_is_final=curr_cls.is_final,
                                   abstract=abstract,
                                   is_interface=curr_cls.is_interface()))
        if signature:
            ret_type, params = self._gen_ret_and_paramas_from_sig(signature)
            funcs.append(
                self.gen_func_decl(ret_type, params=params, not_void=not_void,
                                   class_is_final=curr_cls.is_final,
                                   abstract=abstract,
                                   is_interface=curr_cls.is_interface()))
        if not super_cls_info:
            for _ in range(ut.random.integer(0, max_funcs)):
                funcs.append(
                    self.gen_func_decl(not_void=not_void,
                                       class_is_final=curr_cls.is_final,
                                       abstract=abstract,
                                       is_interface=curr_cls.is_interface()))
        else:
            abstract_funcs = []
            class_decls = self.context.get_classes(self.namespace).values()
            if curr_cls.is_regular():
                abstract_funcs = super_cls_info.super_cls\
                    .get_abstract_functions(class_decls)
                for f in abstract_funcs:
                    funcs.append(
                        self._gen_func_from_existing(
                            f,
                            super_cls_info.type_var_map,
                            curr_cls.is_final,
                            curr_cls.is_interface()
                        )
                    )
                max_funcs = max_funcs - len(abstract_funcs)
            overridable_funcs = super_cls_info.super_cls \
                .get_overridable_functions()
            abstract_funcs = {f.name for f in abstract_funcs}
            overridable_funcs = [f for f in overridable_funcs
                                 if f.name not in abstract_funcs]
            len_over_f = len(overridable_funcs)
            if len_over_f > max_funcs:
                return funcs
            k = ut.random.integer(0, min(max_funcs, len_over_f))
            chosen_funcs = (
                []
                if not max_funcs or curr_cls.is_interface()
                else ut.random.sample(overridable_funcs, k=k)
            )
            for f in chosen_funcs:
                funcs.append(
                    self._gen_func_from_existing(f,
                                                 super_cls_info.type_var_map,
                                                 curr_cls.is_final,
                                                 curr_cls.is_interface()))
            max_funcs = max_funcs - len(chosen_funcs)
            if max_funcs < 0:
                return funcs
            for _ in range(ut.random.integer(0, max_funcs)):
                funcs.append(
                    self.gen_func_decl(not_void=not_void,
                                       class_is_final=curr_cls.is_final,
                                       abstract=abstract,
                                       is_interface=curr_cls.is_interface()))
        return funcs


    # And

    def _gen_func_from_existing(self,
                                func: ast.FunctionDeclaration,
                                type_var_map: tu.TypeVarMap,
                                class_is_final: bool,
                                is_interface: bool) -> ast.FunctionDeclaration:
        """Generate a method that overrides an existing method.

        Args:
            func: Method to override.
            type_var_map: TypeVarMap of func
            class_is_final: is current class final.
            is_interface: is current class an interface.

        Returns:
            A function declaration.
        """
        params = deepcopy(func.params)
        type_params, substituted_type_params = \
            self._gen_type_params_from_existing(func, type_var_map)
        type_param_names = [t.name for t in type_params]
        ret_type = func.ret_type
        for p in params:
            sub = False
            sub_type_map = {
                k: v for k, v in type_var_map.items()
                if k.name not in type_param_names
            }
            old = p.get_type()
            p.param_type = tp.substitute_type(p.get_type(), sub_type_map)
            sub = old != p.get_type()
            if not sub:
                p.param_type = tp.substitute_type(p.get_type(),
                                                  substituted_type_params)
            p.default = None
        sub = False
        sub_type_map = {
            k: v for k, v in type_var_map.items()
            if k.name not in type_param_names
        }
        old = ret_type
        ret_type = tp.substitute_type(ret_type, sub_type_map)
        sub = old != ret_type
        if not sub:
            ret_type = tp.substitute_type(ret_type, substituted_type_params)
        new_func = self.gen_func_decl(func_name=func.name, etype=ret_type,
                                      not_void=False,
                                      class_is_final=class_is_final,
                                      params=params,
                                      is_interface=is_interface,
                                      type_params=type_params)
        if func.body is None:
            new_func.is_final = False
        new_func.override = True
        return new_func

    # Where

    def _gen_type_params_from_existing(self,
                                       func: ast.FunctionDeclaration,
                                       type_var_map
                                      ) -> (List[tp.TypeParameter], tu.TypeVarMap):
        """Gen type parameters for a function that overrides a parameterized
            function.

        Args:
            func: Function to override.
            type_var_map: TypeVarMap of func.

        Returns:
            A list of available type parameters, and TypeVarMap for the type
            parameters of func
        """
        if not func.type_parameters:
            return [], {}
        substituted_type_params = {}
        curr_type_vars = self._get_type_variable_names()
        func_type_vars = [t.name for t in func.type_parameters]
        class_type_vars = [t for t in curr_type_vars
                           if t not in func_type_vars]
        blacklist = func_type_vars + curr_type_vars + list(type_var_map.keys())
        new_type_params = []
        for t_param in func.type_parameters:
            # Here, we substitute the bound of an overridden parameterized
            # function based on the type arguments of the superclass.
            new_type_param = deepcopy(t_param)
            if t_param.name in curr_type_vars:
                # The child class contains a type variable that has the
                # same name with a type variable of the overridden function.
                # So we change the name of the function's type variable to
                # avoid the conflict.
                new_name = ut.random.caps(blacklist=blacklist)
                func_type_vars.append(new_name)
                blacklist.append(new_name)
                new_type_param.name = new_name
                substituted_type_params[t_param] = new_type_param

            if new_type_param.bound is not None:
                sub = False
                sub_type_map = {
                    k: v for k, v in type_var_map.items()
                    if k.name not in func_type_vars \
                    or k.name not in class_type_vars
                }
                old = new_type_param.bound
                bound = tp.substitute_type(new_type_param.bound,
                                           sub_type_map)
                sub = old != bound

                if not sub:
                    bound = tp.substitute_type(bound, substituted_type_params)
                new_type_param.bound = bound
            new_type_params.append(new_type_param)
        return new_type_params, substituted_type_params

    def gen_field_decl(self, etype=None,
                       class_is_final=True,
                       add_to_parent=True) -> ast.FieldDeclaration:
        """Generate a class Field Declaration.

        Args:
            etype: Field type.
            class_is_final: Is the class final.
        """
        name = gu.gen_identifier('lower')
        can_override = not class_is_final and ut.random.bool()
        is_final = ut.random.bool()
        field_type = etype or self.select_type(exclude_contravariants=True,
                                               exclude_covariants=not is_final)
        field = ast.FieldDeclaration(name, field_type, is_final=is_final,
                                     can_override=can_override)
        if add_to_parent:
            log(self.logger, "Adding field {} to context in gen_field_decl".format(field.name))
            self._add_node_to_parent(self.namespace, field)
        return field

    #gen_global_variable_decl added for Rust
    def gen_global_variable_decl(self) -> ast.VariableDeclaration:
        """Generate a global Variable Declaration in Rust.
           Global (static) variable declarations are final, 
           and cannot contain function calls.
           String disabled for now for str String compatibility issues
        """
        #exclude string type for Rust for now, and Vec type
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
        # NOTE maybe we should disable sam coercion for Kotlin
        # the following code does not compile
        # fun interface FI { fun foo(p: Int): Long }
        # var v: FI = {x: Int -> x.toLong()}
        expr = expr or self.generate_expr(var_type, only_leaves,
                                          sam_coercion=True)
        if isinstance(expr, ast.Variable):
            vard = self._get_decl_from_var(expr)
            vard.is_moved = self._move_condition(vard)
            vard.move_prohibited = self._type_moveable(vard)
        self.depth = initial_depth
        is_final = ut.random.bool()
        # We cannot set ? extends X as the type of a variable.
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
        return not decl.get_type().is_primitive() and not (decl.get_type().is_function_type() and decl.get_type().name[0].islower())


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
        '''
        subtypes are irrelevant for now (will be when trait types introduced)
        find_subtype = (
            expr_type and
            subtype and expr_type != self.bt_factory.get_void_type()
            and ut.random.bool()
        )
        
        if find_subtype:
            subtypes = tu.find_subtypes(expr_type, self.get_types(),
                                        include_self=True, concrete_only=True)
            old_type = expr_type
            expr_type = ut.random.choice(subtypes)
            msg = "Found subtype of {}: {}".format(old_type, expr_type)
            log(self.logger, msg)
        '''
        expr_type = expr_type or self.select_type()
        subtype = expr_type
        gen_var = (
            not only_leaves and
            expr_type != self.bt_factory.get_void_type() and
            self._vars_in_context[self.namespace] < cfg.limits.max_var_decls and
            ut.random.bool()
        )
        generators = self.get_generators(expr_type, only_leaves, subtype,
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
            #restricting use of variable in next lines
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
        variables = self._get_assignable_vars()
        initial_depth = self.depth
        self.depth += 1
        ''' commented out for now for Rust
        if not variables:
            # Ok, it's time to find a class with non-final fields,
            # generate an object of this class, and perform the assignment.
            res = self._get_classes_with_assignable_fields()
            if res:
                expr_type, field = res
                variables = [(self.generate_expr(expr_type,
                                                only_leaves, subtype), field)]
        '''
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

    def _get_assignable_vars(self):
        """Get all non-final variables in context that can be assigned.
           If flag_fn or flag_Fn is set to True, then we only consider
           variables from the current namespace, because we do not want
           to capture/mutate variables from outer scopes.
        """
        variables = []
        for var in self.context.get_vars(namespace=self.namespace, only_current=self.flag_fn or self.flag_Fn).values():
            if self.flag_FnMut or self.flag_FnOnce:
                if var.is_moved or var.move_prohibited:
                    continue
            if not getattr(var, 'is_final', True):
                variables.append((None, var))
        return variables

    # And

    def _get_classes_with_assignable_fields(self):
        """Get classes with non-final fields.

        Returns:
            A list that contains tuples of expressions that produce objects
            of a class, and field declarations.
        """
        classes = []
        class_decls = self.context.get_classes(self.namespace).values()
        for c in class_decls:
            for field in c.fields:
                if not field.is_final:
                    classes.append((c, field))
        assignable_types = []
        for c, f in classes:
            t, type_var_map = c.get_type(), {}
            if t.is_type_constructor():
                variance_choices = {
                    t_param: (False, True)
                    for t_param in t.type_parameters
                }
                t, type_var_map = tu.instantiate_type_constructor(
                    t, self.get_types(exclude_arrays=True),
                    variance_choices=variance_choices,
                    disable_variance_functions=self.disable_variance_functions,
                    enable_pecs=self.enable_pecs
                )
                # Ok here we create a new field whose type corresponds
                # to the type argument with which the class 'c' is
                # instantiated.
                f = ast.FieldDeclaration(
                    f.name,
                    field_type=tp.substitute_type(f.get_type(),
                                                  type_var_map)
                )
            assignable_types.append((t, f))

        if not assignable_types:
            return None
        return ut.random.choice(assignable_types)

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

            #if can_wildcard:
             #   variance_choices = gu.init_variance_choices(type_map)
            #else:
            variance_choices = None
            '''s_type, params_map = tu.instantiate_type_constructor(
                s.get_type(),
                self.get_types(),
                type_var_map=type_map,
                enable_pecs=self.enable_pecs,
                disable_variance_functions=self.disable_variance_functions,
                variance_choices=variance_choices,
                disable_variance=variance_choices is None
            )
            '''
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
            '''
            s_type, params_map = tu.instantiate_type_constructor(
                s.get_type(), self.get_types(), type_var_map=type_var_map,
                disable_variance_functions=self.disable_variance_functions)
            '''
            s_type, params_map = self.instantiate_type_constructor(s.get_type(), self.get_types(), type_var_map)
        else:
            s_type, params_map = s.get_type(), {}
            #for now no support for parameterized functions implemented by structs
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
        variables = self.get_variables(self.namespace)
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
        if self.move_semantics and self._type_moveable(varia):
            return True
        return False

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
        arr_len = ut.random.integer(0, 3)
        etype = expr_type.type_args[0]
        exprs = [
            self.generate_expr(etype, only_leaves=only_leaves, subtype=subtype)
            for _ in range(arr_len)
        ]
        # An array expression (i.e., emptyArray<T>(), arrayOf<T>) cannot
        # take wildcards.
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
        exclude_function_types = self.language == 'java' or self.language == 'rust'
        etype = self.select_type(exclude_function_types=exclude_function_types, exclude_type_vars=True, exclude_usr_types=True)
        op = ut.random.choice(ast.EqualityExpr.VALID_OPERATORS[self.language])
        e1 = self.generate_expr(etype, only_leaves, subtype=False)
        e2 = self.generate_expr(etype, only_leaves, subtype=False)
        self.depth = initial_depth
        self.move_semantics = prev_move_semantics
        return ast.EqualityExpr(e1, e2, op)

    # pylint: disable=unused-argument
    def gen_logical_expr(self,
                         expr_type=None,
                         only_leaves=False) -> ast.LogicalExpr:
        """Generate a logical expression

        It generates two additional expression for the logical expression.

        Args:
            expr_type: exists for compatibility reasons.
            only_leaves: do not generate new leaves except from `expr`.
        """
        initial_depth = self.depth
        self.depth += 1
        op = ut.random.choice(ast.LogicalExpr.VALID_OPERATORS[self.language])
        e1 = self.generate_expr(self.bt_factory.get_boolean_type(),
                                only_leaves)
        e2 = self.generate_expr(self.bt_factory.get_boolean_type(),
                                only_leaves)
        self.depth = initial_depth
        return ast.LogicalExpr(e1, e2, op)

    # pylint: disable=unused-argument
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
        if self.language == 'java' and e1_type.name in ('Boolean', 'String'):
            op = ut.random.choice(
                ast.EqualityExpr.VALID_OPERATORS[self.language])
            return ast.EqualityExpr(e1, e2, op)
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
        subtype = False #subtypes irrelevant for now
        if subtype:
            subtypes = tu.find_subtypes(etype, self.get_types(),
                                        include_self=True, concrete_only=True)
            true_type = ut.random.choice(subtypes)
            false_type = ut.random.choice(subtypes)
            tmp_t = ut.random.choice(subtypes)
            # Find which of the given types is the supertype.
            cond_type = functools.reduce(
                lambda acc, x: acc if x.is_subtype(acc) else x,
                [true_type, false_type],
                tmp_t
            )
        else:
            true_type, false_type, cond_type = etype, etype, etype
        true_expr = ast.Block([self.generate_expr(true_type, only_leaves, subtype=False)], is_func_block = False) #TODO change this, enclosed in block only for Rust
        false_expr = ast.Block([self.generate_expr(false_type, only_leaves, subtype=False)], is_func_block = False) #TODO change this, enclosed in block only for Rust
        self.depth = initial_depth
        self.move_semantics = prev_move_semantics
        # Note that this an approximation of the type of the whole conditional.
        # To properly estimate the type of conditional, we need to implement
        # the LUB algorithm.
        # Note the type passed in conditional may be imprecise in the following
        # scenario:
        # class A
        # class B extends A
        # class C extends B
        # class D extends B
        #
        # gen_conditional with type A
        # true branch type C
        # false branch type D
        #
        # The type will assign to the conditional will be A, but the correct
        # one is B.
        return ast.Conditional(cond, true_expr, false_expr, cond_type)

    def gen_is_expr(self,
                    expr_type: tp.Type,
                    only_leaves=False,
                    subtype=True) -> ast.Conditional:
        """Generate an is expression.

        If it cannot detect a subtype for the expr_type, then it just generates
        a new expression of expr_type.

        Args:
            expr_type: type to smart cast.
            only_leaves: do not generate new leaves except from `expr`.
            subtype: The type of the sub expressions could be a subtype of
                `expr_type`.

        Returns:
            A conditional with is.
        """
        def _get_extra_decls(namespace):
            return [
                v
                for v in self.context.get_declarations(
                    namespace, only_current=True).values()
                if (isinstance(v, (ast.VariableDeclaration,
                                  ast.FunctionDeclaration)))
            ]

        final_vars = [
            v
            for v in self.context.get_vars(self.namespace).values()
            if (
                # We can smart cast local variables that are final, have
                # explicit types, and are not overridable.
                isinstance(v, ast.VariableDeclaration) and
                getattr(v, 'is_final', True) and
                not v.is_type_inferred and
                not getattr(v, 'can_override', True)
            )
        ]
        if not final_vars:
            return self.generate_expr(expr_type, only_leaves=True,
                                      subtype=subtype)
        prev_depth = self.depth
        self.depth += 3
        var = ut.random.choice(final_vars)
        var_type = var.get_type()
        subtypes = tu.find_subtypes(var_type, self.get_types(),
                                    include_self=False, concrete_only=True)
        subtypes = self._filter_subtypes(subtypes, var_type)
        if not subtypes:
            return self.generate_expr(expr_type, only_leaves=True,
                                      subtype=subtype)

        subtype = ut.random.choice(subtypes)
        initial_decls = _get_extra_decls(self.namespace)
        prev_namespace = self.namespace
        self.namespace += ('true_block',)
        # Here, we create a 'virtual' variable declaration inside the
        # namespace of the block corresponding to the true branch. This
        # variable has the same name with the variable that appears in
        # the left-hand side of the 'is' expression, but its type is the
        # selected subtype.
        log(self.logger, "Adding variable {} to context in gen_is_expr".format(var.name))
        self.context.add_var(self.namespace, var.name,
            ast.VariableDeclaration(
                var.name,
                ast.BottomConstant(var.get_type()),
                var_type=subtype))
        true_expr = ast.Block(self.generate_expr(expr_type), is_func_block = False) #TODO change this, enclosed in block only for Rust  true_expr = self.generate_expr(expr_type) 
        # We pop the variable from context. Because it's no longer used.
        self.context.remove_var(self.namespace, var.name)
        extra_decls_true = [v for v in _get_extra_decls(self.namespace)
                            if v not in initial_decls]
        if extra_decls_true:
            true_expr = ast.Block(extra_decls_true + [true_expr],
                                  is_func_block=False)
        self.namespace = prev_namespace + ('false_block',)
        false_expr = ast.Block([self.generate_expr(expr_type, only_leaves=only_leaves,
                                        subtype=subtype)], is_func_block = False) #TODO change this, enclosed in block only for Rust
        extra_decls_false = [v for v in _get_extra_decls(self.namespace)
                             if v not in initial_decls]
        if extra_decls_false:
            false_expr = ast.Block(extra_decls_false + [false_expr],
                                   is_func_block=False)
        self.namespace = prev_namespace
        self.depth = prev_depth
        return ast.Conditional(
            ast.Is(ast.Variable(var.name), subtype),
            true_expr,
            false_expr,
            expr_type
        )

    # Where

    def _filter_subtypes(self, subtypes, initial_type):
        """Filter out types that cannot be smart casted.

        The types that cannot be smart casted are Type Variables and
        Parameterized Types. The only exception is Kotlin in which we can
        smart cast parameterized types.
        """
        new_subtypes = []
        for t in subtypes:
            if t.is_type_var():
                continue
            if self.language != 'kotlin':
                # We can't check the instance of a parameterized type due
                # to type erasure. The only exception is Kotlin, see below.
                if not t.is_parameterized():
                    new_subtypes.append(t)
                continue

            # In Kotlin, you can smart cast a parameterized type like the
            # following.

            # class A<T>
            # class B<T> extends A<T>
            # fun test(x: A<String>) {
            #   if (x is B) {
            #      // the type of x is B<String> here.
            #   }
            # }
            if t.is_parameterized():
                t_con = t.t_constructor
                if t_con.is_subtype(initial_type):
                    continue
            new_subtypes.append(t)
        return new_subtypes


    def gen_lambda(self,
                   etype: tp.Type=None,
                   not_void=False,
                   params: List[ast.ParameterDeclaration]=None,
                   only_leaves=False,
                   signature: tp.Type=None,
                  ) -> ast.Lambda:
        """Generate a lambda expression.

        Lambdas have shadow names that we can use them in the context to
        retrieve them.

        Args:
            etype: return type of the lambda.
            not_void: the lambda should not return void.
            shadow_name: give a specific shadow name.
            params: parameters for the lambda.
            signature: generate lambda with a given signature.
        """
        prev_flag_fn = self.flag_fn
        prev_flag_Fn = self.flag_Fn
        prev_flag_FnMut = self.flag_FnMut
        prev_flag_FnOnce = self.flag_FnOnce
        if signature is not None:
            if signature.name == "fn":
                self.flag_fn = True
            elif signature.name == "Fn":
                self.flag_Fn = True
            elif signature.name == "FnMut":
                self.flag_FnMut = True
            elif signature.name == "FnOnce":
                self.flag_FnOnce = True

        if self.declaration_namespace:
            namespace = self.declaration_namespace
        else:
            namespace = self.namespace

        initial_namespace = self.namespace
        shadow_name = "lambda_" + str(next(self.int_stream))
        self.namespace += (shadow_name,)
        initial_depth = self.depth
        self.depth += 1

        params = params if params is not None else self._gen_func_params(for_lambda=True)
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
        self.flag_fn = prev_flag_fn
        self.flag_Fn = prev_flag_Fn
        self.flag_FnMut = prev_flag_FnMut
        self.flag_FnOnce = prev_flag_FnOnce
        return res

    def gen_func_call(self,
                      etype: tp.Type,
                      only_leaves=False,
                      subtype=True) -> ast.FunctionCall:
        """Generate a function call.

        The function call could be either a normal function call, or a function
        call from a function reference.
        Note that this function may generate a new function/class as a side
        effect.

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
            # NOTE we could use _gen_func_call to generate function references
            # for producing function calls, but then we should always cast them.
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
                    '''
                    type_fun_rec, _ = tu.instantiate_type_constructor(
                        type_fun.receiver_t, self.get_types(),
                        type_var_map=type_fun.receiver_inst,)
                    '''
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
            if not param.vararg:
                arg = self.generate_expr(expr_type, only_leaves,
                                         gen_bottom=gen_bottom)
                if param.default:
                    if self.language in ['kotlin', 'scala'] and ut.random.bool():
                        # Randomly skip some default arguments.
                        args.append(ast.CallArgument(arg, name=param.name))
                else:
                    args.append(ast.CallArgument(arg))

            else:
                # This param is a vararg, so provide a random number of
                # arguments.
                for _ in range(ut.random.integer(0, 3)):
                    args.append(ast.CallArgument(
                        self.generate_expr(
                            expr_type.type_args[0],
                            only_leaves,
                            gen_bottom=gen_bottom)))
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
        variables = self.get_variables(self.namespace)
        for var in variables:
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
        variables = self.get_variables(self.namespace)
        for var in variables:
            var_name = var.name
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
    
    '''
    def _concretize_type(self, t, prev_type_map):
        """ Replace abstract trait type with a matching concrete type implementing this trait 
            By design of creating trait bounds, there should exist a matching struct
        """
        if t in prev_type_map:
            #type parameter has been concretized already
            return prev_type_map[t]
        if t.is_type_var() and t.bound is not None:
            trait_type = deepcopy(t.bound)
            if trait_type.is_parameterized():
                trait_type.type_args = [self._concretize_type(t_arg, prev_type_map) for t_arg in trait_type.type_args]
            for s_name, lst in self._impls.items():
                for (impl, s_map, t_map) in lst:
                    impl_type_map = {t_param: self.select_type() for t_param in impl.type_parameters}
                    if not impl.trait.has_type_variables():
                        if impl.trait == trait_type:
                            updated_type = tp.substitute_type(impl.struct, impl_type_map)
                            return updated_type
                    else:
                        trait_map = tu.unify_types(trait_type, impl.trait, self.bt_factory, same_type=False)
                        if not trait_map:
                            continue
                        impl_type_map.update(trait_map)
                        updated_type = tp.substitute_type(impl.struct, impl_type_map)
                        return updated_type
        if t.is_parameterized():
            updated_type_args = []
            for t_arg in t.type_args:
                updated_type_args.append(self._concretize_type(t_arg, prev_type_map))
            updated_type = deepcopy(t)
            updated_type.type_agrs = updated_type_args
            return updated_type
        return t
    '''

    # pylint: disable=unused-argument
    def gen_new(self,
                etype: tp.Type,
                only_leaves=False,
                subtype=True,
                sam_coercion=False) -> ast.New:
        """Create a new object of a given type.

        This could be:
            * Function type
            * SAM type
            * Parameterized Type
            * Simple Classifier Type

        Args:
            etype: the type for which we want to create an object
            only_leaves: do not generate new leaves except from `expr`.
            subtype: The type could be a subtype of `etype`.
            sam_coercion: Apply sam coercion if possible.
        """
        if getattr(etype, 'is_function_type', lambda: False)():
            return self._gen_func_ref_lambda(etype, only_leaves=only_leaves)
        # Apply SAM coercion
        if (sam_coercion and tu.is_sam(self.context, etype)
                and ut.random.bool(cfg.prob.sam_coercion)):
            type_var_map = tu.get_type_var_map_from_ptype(etype)
            sam_sig_etype = tu.find_sam_fun_signature(
                    self.context,
                    etype,
                    self.bt_factory.get_function_type,
                    type_var_map=type_var_map
            )
            if sam_sig_etype:
                return self._gen_func_ref_lambda(sam_sig_etype,
                                                 only_leaves=only_leaves)
        
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
        
        #instantiating a new struct in Rust
        #if self.language == 'rust' and isinstance(etype, tp.SimpleClassifier):
        if s_decl is None or etype.name in self._blacklisted_structs:
            t = etype
            # If the etype corresponds to a type variable not belonging to
            # to the current namespace, then create a bottom constant
            # whose type is unknown. This means that the corresponding
            # translator won't perform cast on this constant.
            if etype.is_type_var() and (
                    etype.name not in self._get_type_variable_names()):
                t = None
            return ast.BottomConstant(t)
        
        if etype.is_type_constructor():
            '''
            etype, _ = tu.instantiate_type_constructor(
                etype, self.get_types(),
                disable_variance_functions=self.disable_variance_functions,
                enable_pecs=self.enable_pecs)
            '''
            etype, _ = self.instantiate_type_constructor(etype, self.get_types())
        if s_decl.is_parameterized() and (
              s_decl.get_type().name != etype.name):
            '''etype, _ = tu.instantiate_type_constructor(
                s_decl.get_type(), self.get_types(),
                disable_variance_functions=self.disable_variance_functions,
                enable_pecs=self.enable_pecs)
            '''
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
            # Generate a bottom value, if we are in this case:
            # class A(val x: A)
            # Generating a bottom constants prevents us from infinite loops.
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
        #return ast.New(new_type, args)

    # Where

    def _get_subclass(self,
                      etype: tp.Type,
                      subtype=True) -> ast.ClassDeclaration:
        """"Find a subclass that is a subtype of the given type and is a
        regular class.

        Args:
            etype: the type for which we are searching for subclasses.
            subtype: The type could be a subtype of `etype`.
        """
        class_decls = self.context.get_classes(self.namespace).values()
        # Get all classes that are subtype of the given type, and there
        # are regular classes (no interfaces or abstract classes).
        subclasses = []
        for c in class_decls:
            if c.class_type != ast.ClassDeclaration.REGULAR:
                continue
            if c.is_parameterized():
                t_con = getattr(etype, 't_constructor', None)
                if c.get_type() == t_con or (
                        subtype and c.get_type().is_subtype(etype)):
                    subclasses.append(c)
            else:
                if c.get_type() == etype or (
                        subtype and c.get_type().is_subtype(etype)):
                    subclasses.append(c)
        if not subclasses:
            return None
        # FIXME what happens if subclasses is empty?
        # it may happens due to ParameterizedType with TypeParameters as targs
        return ut.random.choice(
            [s for s in subclasses if s.name == etype.name] or subclasses)

    # And

    def _gen_func_ref_lambda(self, etype:tp.Type, only_leaves=False):
        """Generate a function reference or a lambda for a given signature.

        Args:
            etype: signature

        Returns:
            ast.Lambda or ast.FunctionReference
        """
        # We are unable to produce function references in super calls.
        if etype.name[0].islower(): #temporary, might change this ut.random.bool(cfg.prob.func_ref) or 
            func_ref = self._gen_func_ref(etype, only_leaves=only_leaves)
            if func_ref:
                return func_ref

        #DISABLING Lambdas for now for Rust!!!! Change this later
        #func_ref = self._gen_func_ref(etype, only_leaves=only_leaves)
        #if func_ref:
        #    return func_ref

        # Generate Lambda
        ret_type, params = self._gen_ret_and_paramas_from_sig(etype, True)
        return self.gen_lambda(etype=ret_type, params=params,
                               only_leaves=only_leaves, signature=etype)

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

        if self.language == "rust" and len(self.namespace) > 3 and self.namespace[1][0].isupper() and self.namespace[-2][0].islower():
            other_candidates.remove(gen_fun_call) #removing function calls for inner trait functions for Rust
            if expr_type == self.bt_factory.get_void_type():
                return [lambda x: ast.BottomConstant(x)]

        if expr_type == self.bt_factory.get_void_type():
            # The assignment operator in Java evaluates to the assigned value.
            #if self.language == 'java':
            #    return [gen_fun_call]
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
                  exclude_usr_types=False,
                  include_func_traits=False) -> List[tp.Type]:
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
            include_func_traits: include function traits: Fn, FnMut, FnOnce.

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
                if self.flag_fn else self.context.get_types(self.namespace)
            for t_param in t_params.values():
                variance = getattr(t_param, 'variance', None)
                if exclude_covariants and variance == tp.Covariant:
                    continue
                if exclude_contravariants and variance == tp.Contravariant:
                    continue
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
                    exclude_usr_types=False,
                    include_fn_traits=False) -> tp.Type:
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
        if stype.name=="fn" and ut.random.bool():
            #If function type is selected, decide if we want to select some function trait type instead.
            stype = ut.random.choice(self.function_trait_types)
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
        variances = [tp.Invariant, tp.Covariant, tp.Contravariant]
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
                # OK we do this trick for type parameters corresponding to
                # functions in order to avoid conflicts with type variables
                # of classes. TODO: consider being less conservative.
                name = "F_" + name
            if for_impl:
                name = "I_" + name
            variance = None
            if with_variance and ut.random.bool():
                variance = ut.random.choice(variances)
            bound = None

            if True: #ut.random.bool(cfg.prob.bounded_type_parameters):
                exclude_covariants = variance == tp.Contravariant or for_function
                exclude_contravariants = True
                bound = self.choose_trait_bound()
            type_param = tp.TypeParameter(name, variance=variance, bound=bound)
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
        fn_trait_constr = ut.random.choice(self.function_trait_types)
        fn_trait, _ = self.instantiate_type_constructor(fn_trait_constr, self.get_types())
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
    ### Internal helper functions ###

    def _get_type_variable_names(self) -> List[str]:
        """Get the name of type variables that are in place in the current
        namespace.
        """
        return list(self.context.get_types(self.namespace).keys())

    def _get_func_ret_type(self,
                           params: List[ast.ParameterDeclaration],
                           etype: tp.Type,
                           not_void=False) -> tp.Type:
        """Get return type for a function or lambda.

        Args:
            params: function parameters.
            etype: use this type as the return type
            not_void: do not return void
        """
        if etype is not None:
            return etype
        param_types = [p.param_type for p in params
                       if getattr(p.param_type,
                                  'variance', None) != tp.Contravariant]
        if param_types and ut.random.bool():
            return ut.random.choice(param_types)
        ret = self.select_type(exclude_contravariants=True)
        return ret

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

    def _get_class(self,
                   etype: tp.Type
                  ) -> Tuple[ast.ClassDeclaration, tu.TypeVarMap]:
        """Find the class declaration for a given type.
        """
        # Get class declaration based on the given type.
        class_decls = self.context.get_classes(self.namespace).values()
        for c in class_decls:
            cls_type = c.get_type()
            t_con = getattr(etype, 't_constructor', None)
            # or t == t_con: If etype is a parameterized type (i.e.,
            # getattr(etype, 't_constructor', None) != None), we need to
            # get the class corresponding to its type constructor.
            if cls_type.name == etype.name or cls_type == t_con:
                if c.is_parameterized():
                    type_var_map = {
                        t_param: etype.type_args[i]
                        for i, t_param in enumerate(c.type_parameters)
                    }
                else:
                    type_var_map = {}
                return c, type_var_map
        return None

    def _get_var_type_to_search(self, var_type: tp.Type) -> tp.TypeParameter:
        """Get the type that we want to search for.

        We exclude:
            * built-ins
            * type variables/wildcards without bounds
            * type variables/wildcards with bounds to a type variable

        Args:
            var_type: The type of the variable.

        Returns:
            var_type or None
        """
        # We are only interested in variables of class types.
        if tu.is_builtin(var_type, self.bt_factory) or var_type.is_type_var():
            return None
        '''
        if var_type.is_type_var() or var_type.is_wildcard():
            args = [] if var_type.is_wildcard() else [self.bt_factory]
            bound = var_type.get_bound_rec(*args)
            if not bound or tu.is_builtin(bound, self.bt_factory) or (
                  isinstance(bound, tp.TypeParameter)):
                return None
            var_type = bound
        '''
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
        variables = self.get_variables(self.namespace)
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

    # helper generators

    def _gen_func_params(self, for_lambda=False) -> List[ast.ParameterDeclaration]:
        """Generate parameters for a function or for a lambda.
        """
        params = []
        for i in range(ut.random.integer(0, cfg.limits.fn.max_params)):
            param = self.gen_param_decl(for_lambda=for_lambda)
            params.append(param)
        return params

    # Where

    def _can_vararg_param(self, param: ast.ParameterDeclaration) -> bool:
        """Check if a parameter can be vararg.
        """
        if self.language == 'kotlin':
            # TODO theosotr Can we do this in a better way? without hardcode?
            # Actually in Kotlin, the type of varargs is Array<out T>.
            # So, until we add support for use-site variance, we support
            # varargs for 'primitive' types only which kotlinc treats them
            # as specialized arrays.
            t_constructor = getattr(param.get_type(), 't_constructor', None)
            return isinstance(t_constructor, kt.SpecializedArrayType)
        elif self.language == "scala":
            return param.get_type().name == "Seq"
        else:
            # A vararg is actually a syntactic sugar for a parameter whose type
            # is an array of something.
            return param.get_type().name == 'Array'

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
            # The function does not contain any declarations and its return
            # type is not Unit. So, we can create an expression-based function.
            #body = expr if ut.random.bool(cfg.prob.function_expr) else \
             #   ast.Block([expr])
            body = ast.Block([expr]) #enclosed in block for Rust
        else:
            exprs, decls = self._gen_side_effects()
            body = ast.Block(decls + exprs + [expr])
        return body

    # Where

    def _gen_side_effects(self) -> Tuple[List[ast.Expr], List[ast.Declaration]]:
        """Generate expressions with side-effects for function bodies.

        Example side-effects: assignment, variable declaration, etc.
        """
        exprs = []
        for _ in range(ut.random.integer(0, cfg.limits.fn.max_side_effects)):
            expr = self.generate_expr(self.bt_factory.get_void_type())
            if expr:
                exprs.append(expr)
        # These are the new declarations that we created as part of the side-
        # effects.
        decls = self.context.get_declarations(self.namespace, True).values()
        decls = [d for d in decls
                 if not isinstance(d, ast.ParameterDeclaration)]
        return exprs, decls

    def _gen_ret_and_paramas_from_sig(self, etype, inside_lambda=False) -> \
            Tuple[tp.Type, ast.ParameterDeclaration]:
        """Generate parameters from signature and return them along with return
        type.

        Args:
            etype: signature type
            inside_lambda: true if we want to generate parameters for a lambda
        """
        if inside_lambda:
            prev_inside_java_lamdba = self._inside_java_lambda
            self._inside_java_lambda = self.language == "java"
        params = [self.gen_param_decl(et) for et in etype.type_args[:-1]]
        if inside_lambda:
            self._inside_java_lambda = prev_inside_java_lamdba
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

        This function essentially searches for variables containing objects
        whose class has either a field of a specific value or a function that
        return a particular value.

        As for func_ref and signatures there are the following scenarios:

        1. func_ref = True and signature = False and attr_name = fields
            -> find function references that return etype
        2. func_ref = False and signature = True and attr_name = functions
            -> find functions that have the given signature
        2. func_ref = True and signature = True and attr_name = fields
            -> find functions references that return etype (etype is signature)

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
        variables = self.get_variables(self.namespace)
        for var in variables:
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
                
                #Handling of function calls on structs
                if attr_name == 'functions':
                    fun_type_var_map = {}
                    if attr.is_parameterized():
                        raise ValueError("Calls of parameterized functions on structs are not supported yet")
                    else:
                        fun_type_var_map = {}
                    type_var_map.update(fun_type_var_map)

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
            for f in impl.functions:
                updated_f = self._update_func_decl(f, updated_map)
                funcs.append(updated_f)
            '''
            if not impl.type_parameters:
                if s_map == type_var_map:
                    funcs.extend(impl.functions)
            else:
                is_compatible = True
                impl_type_map = {}
                for key in s_map.keys():
                    curr_inst = s_map[key]
                    #Fix this: conflicting instantiations (I, I and i32, f64)
                    if not curr_inst in impl.type_parameters and curr_inst != type_var_map[key]:
                        is_compatible = False
                        break
                    if curr_inst in impl.type_parameters:
                        impl_type_map[curr_inst] = type_var_map[key]
                if is_compatible:
                    for f in impl.functions:
                        updated_f = self._update_func_decl(f, impl_type_map)
                        funcs.append(updated_f)
            if s_map == type_var_map:
                curr_funcs = t_decl.function_signatures + t_decl.default_impls
                updated_funcs = []
                for f in curr_funcs:
                    updated_funcs.append(self._update_func_decl(f, t_map))
                funcs += updated_funcs
            '''
        return funcs

    def _is_impl_compatible(self, impl_map, type_var_map):
        updated_map = {}
        for key in impl_map.keys():
            curr_inst = impl_map[key]
            if not curr_inst.has_type_variables():
                if curr_inst != type_var_map[key]:
                    return False, {}
            else:
                #type_map = tu.unify_types(type_var_map[key], self._erase_bounds(curr_inst), self.bt_factory, same_type=False)
                type_map = self.unify_types(type_var_map[key], curr_inst)
                if not type_map:# or not self._map_fulfills_bounds(type_map):
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
        `etype`, and then it also searches for receivers whose class has a
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
        # First find all top-level functions or methods included
        # in the current class.
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
            if func.is_parameterized() and func.is_class_method():
                # TODO: Consider being less conservative.
                # The problem is when the class method is parameterized,
                # the receiver is parameterized, and the type parameters
                # of functions have bounds corresponding to the type parameters
                # of class.
                continue

            type_var_map = {}
            if func.is_parameterized():
                func_type_var_map = tu.unify_types(etype, func.get_type(),
                                                   self.bt_factory)
                if not func_type_var_map:
                    continue
                '''
                func_type_var_map = tu.instantiate_parameterized_function(
                    func.type_parameters, self.get_types(),
                    type_var_map=func_type_var_map, only_regular=True
                )
                '''
                func_type_var_map = self.instantiate_parameterized_function(func.type_parameters, self.get_types(), func_type_var_map)
                type_var_map.update(func_type_var_map)

            if not self._is_sigtype_compatible(func, etype, type_var_map,
                                               signature, subtype):
                continue

            # Nice to have:  add `this` explicitly as the receiver in methods
            # of current class.
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
                for f in impl.functions:
                    if f.get_type() == self.bt_factory.get_void_type():
                        continue
                    if not f.get_type().has_type_variables() or etype == self.bt_factory.get_void_type():
                        if f.get_type() != etype: #types do not match
                            continue
                    else:
                        #type_map = tu.unify_types(etype, f.get_type(), self.bt_factory, same_type=False)
                        type_map = self.unify_types(etype, f.get_type())
                        if not type_map:# or not self._map_fulfills_bounds(type_map): #types are not compatible
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
        """ Generate a function or a class containing a function whose return
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
            not etype.is_type_var() and #functions returning parameterized types are not yet supported for trait/struct functions
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
                '''
                func_type_var_map = tu.instantiate_parameterized_function(
                    func.type_parameters, self.get_types(),
                    only_regular=True, type_var_map={})
                '''
                func_type_var_map = self.instantiate_parameterized_function(func.type_parameters, self.get_types(), {})
                func_ret_type = func.get_type()
                mapping = tu.unify_types(func_ret_type, etype, self.bt_factory)
                func_type_var_map.update(mapping) #assigning correct type annotation for parameterized function (for Rust)
                
            msg = "Generating a method {} of type {}; TypeVarMap {}".format(
                func.name, etype, func_type_var_map)
            log(self.logger, msg)
            return gu.AttrAccessInfo(None, {}, func, func_type_var_map)
        #Only non-parameterized types are supported
        return self.gen_matching_impl(etype)

    def _get_matching_class(self,
                            etype: tp.Type,
                            subtype: bool,
                            attr_name: str,
                            signature=False) -> gu.AttrAccessInfo:
        """Get a class that has an attribute of attr_name that is/return etype.

        This function essentially searches for a class that has either a field
        of a specific value or a function that return a particular value.

        Args:
            etype: the targeted type that we are searching. Functions should
                return that type.
            subtype: The type of matching attribute could be a subtype of
                `etype`.
            attr_name: 'fields' or 'functions'
            signature: etype is a signature.

        Returns:
            An AttrAccessInfo with a matched class type and attribute
            declaration (field or function).
        """
        msg = "Searching for class that contains {} of type {}"
        log(self.logger, msg.format(attr_name, etype))
        class_decls = self._get_matching_class_decls(
            etype, subtype=subtype, attr_name=attr_name, signature=signature)
        if not class_decls:
            return None
        cls, type_var_map, attr = ut.random.choice(class_decls)
        func_type_var_map = {}
        is_parameterized_func = isinstance(
            attr, ast.FunctionDeclaration) and attr.is_parameterized()
        if cls.is_parameterized():
            cls_type_var_map = type_var_map

            variance_choices = (
                None
                if cls_type_var_map is None
                else gu.init_variance_choices(cls_type_var_map)
            )
            cls_type, params_map = tu.instantiate_type_constructor(
                cls.get_type(), self.get_types(),
                only_regular=True, type_var_map=type_var_map,
                enable_pecs=self.enable_pecs,
                disable_variance_functions=self.disable_variance_functions,
                variance_choices=variance_choices,
                disable_variance=variance_choices is None
            )
            msg = ("Found parameterized class {} with TypeVarMap {} and "
                   "incomplete TypeVarMap {}")
            log(self.logger, msg.format(cls.name, params_map, type_var_map))
            if is_parameterized_func:
                # Here we have found a parameterized function in a
                # parameterized class. So wee need to both instantiate
                # the type constructor and the parameterized function.
                types = tu._get_available_types(cls.get_type(),
                                                self.get_types(),
                                                True, False)
                _, type_var_map = tu._compute_type_variable_assignments(
                    cls.type_parameters + attr.type_parameters,
                    types, type_var_map=type_var_map,
                    variance_choices=variance_choices
                )
                params_map, func_type_var_map = tu.split_type_var_map(
                    type_var_map, cls.type_parameters, attr.type_parameters)
                targs = [
                    params_map[t_param]
                    for t_param in cls.type_parameters
                ]
                cls_type = cls.get_type().new(targs)
            else:
                # Here, we have a non-parameterized function in a parameterized
                # class. So we only need to instantiate the type constructor.
                cls_type, params_map = tu.instantiate_type_constructor(
                    cls.get_type(), self.get_types(),
                    only_regular=True, type_var_map=cls_type_var_map,
                    enable_pecs=self.enable_pecs,
                    variance_choices=variance_choices,
                    disable_variance=variance_choices is None
                )
        else:
            if is_parameterized_func:
                # We are in a parameterized class defined in a class that
                # is not a type constructor.
                func_type_var_map = tu.instantiate_parameterized_function(
                    attr.type_parameters, self.get_types(),
                    only_regular=True, type_var_map=type_var_map)
            cls_type, params_map = cls.get_type(), {}

        attr_msg = "Attribute {}; type: {}, TypeVarMap{}".format(
            attr_name, etype, func_type_var_map)
        msg = "Selected class {} with TypeVarMap {};" " matches {}".format(
            cls.name, params_map, attr_msg)
        log(self.logger, msg)
        return gu.AttrAccessInfo(cls_type, params_map, attr, func_type_var_map)

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

    def _get_matching_class_decls(self,
                                  etype: tp.Type,
                                  subtype: bool,
                                  attr_name: str,
                                  signature=False
                                 ) -> List[Tuple[ast.ClassDeclaration,
                                                 tu.TypeVarMap,
                                                 ast.Declaration]]:
        """Get classes that have attributes of attr_name that are/return etype.

        Args:
            etype: the targeted type that we are searching. Functions should
                return that type.
            subtype: The type of matching attribute could be a subtype of
                `etype`.
            attr_name: 'fields' or 'functions'
            signature: etype is a signature.

        Returns:
            A list of tuples that include class declarations, TypeVarMaps for
            the attributes and the declarations of the attributes (fields or
            functions).
        """

        class_decls = []
        for c in self.context.get_classes(self.namespace).values():
            for attr in self._get_class_attributes(c, attr_name):
                attr_type = attr.get_type()
                if not attr_type:
                    continue
                if attr_type == self.bt_factory.get_void_type():
                    continue
                # Avoid recursive decls because of incomplete information.
                if attr.name == self.namespace[-1] and signature:
                    continue

                is_comb, type_var_map = self._is_signature_compatible(
                    attr, etype, signature, subtype)
                if not is_comb:
                    continue
                # Now here we keep the class and the function that match
                # the given type.
                class_decls.append((c, type_var_map, attr))
        return class_decls

    def _gen_matching_class(self,
                            etype: tp.Type,
                            attr_name: str,
                            not_void=False,
                            signature=False) -> gu.AttrAccessInfo:
        """Generate a class that has an attribute of attr_name that is/return etype.

        Args:
            etype: the targeted type that we want to get. Functions should
                return that type.
            attr_name: 'fields' or 'functions'
            not_void: Functions of the class should not return void.
            signature: etype is a signature.

        Returns:
            An AttrAccessInfo for the generated class type and attribute
            declaration (field or function).
        """
        initial_namespace = self.namespace
        class_name = gu.gen_identifier('capitalize')
        type_params = None

        # Get return type, type_var_map, and flag for wildcards
        if etype.has_type_variables():
            # We have to create a class that has an attribute whose type
            # is a type parameter. The only way to achieve this is to create
            # a parameterized class, and pass the type parameter 'etype'
            # as a type argument to the corresponding type constructor.
            self.namespace = ast.GLOBAL_NAMESPACE + (class_name,)
            type_params, type_var_map, can_wildcard = \
                self._create_type_params_from_etype(etype)
            etype2 = tp.substitute_type(etype, type_var_map)
        else:
            type_var_map, etype2, can_wildcard = {}, etype, False

        self.namespace = ast.GLOBAL_NAMESPACE

        # Create class
        if attr_name == 'functions':
            kwargs = {'fret_type': etype2} if not signature \
                else {'signature': etype2}
        else:
            kwargs = {'field_type': etype2}
        cls = self.gen_class_decl(**kwargs, not_void=not_void,
                                  type_params=type_params,
                                  class_name=class_name)
        self.namespace = initial_namespace

        # Get receiver
        if cls.is_parameterized():
            type_map = {v: k for k, v in type_var_map.items()}
            if etype2.is_primitive() and (
                    etype2.box_type() == self.bt_factory.get_void_type()):
                type_map = None

            if can_wildcard:
                variance_choices = gu.init_variance_choices(type_map)
            else:
                variance_choices = None
            cls_type, params_map = tu.instantiate_type_constructor(
                cls.get_type(),
                self.get_types(),
                type_var_map=type_map,
                enable_pecs=self.enable_pecs,
                disable_variance_functions=self.disable_variance_functions,
                variance_choices=variance_choices,
                disable_variance=variance_choices is None
            )
        else:
            cls_type, params_map = cls.get_type(), {}

        # Generate func_type_var_map
        for attr in getattr(cls, attr_name):
            if not self._is_sigtype_compatible(attr, etype, params_map,
                                               signature, False):
                continue

            func_type_var_map = {}
            if isinstance(
                    attr, ast.FunctionDeclaration) and attr.is_parameterized():
                func_type_var_map = tu.instantiate_parameterized_function(
                    attr.type_parameters, self.get_types(), only_regular=True,
                    type_var_map=params_map)

            msg = ("Generated a class {} with an attribute {} of type {}; "
                   "ClassTypeVarMap {}, FuncTypeVarMap {}")
            log(self.logger, msg.format(cls.name, attr_name, etype,
                                        params_map, func_type_var_map))
            return gu.AttrAccessInfo(cls_type, params_map, attr,
                                     func_type_var_map)
        return None

    # Where

    def _create_type_params_from_etype(self, etype: tp.Type):
        """Generate type parameters for a type.

        Returns:
            * A list of type parameters.
            * A TypeVarMap for the type parameters.
            * A boolean to declare if we can use wildcards.
        """
        if not etype.has_type_variables():
            return []
        '''
        #added for now, probably wrong
        if isinstance(etype, tp.TypeConstructor):
            type_map = {k: k for k in etype.type_parameters}
            return etype.type_parameters, type_map, False
        '''
        if isinstance(etype, tp.TypeParameter):
            type_params = self.gen_type_params(count=1)
            type_params[0].bound = etype.get_bound_rec(self.bt_factory)
            type_params[0].variance = tp.Invariant
            return type_params, {etype: type_params[0]}, True
        
        # the given type is parameterized
        #assert isinstance(etype, (tp.ParameterizedType, tp.WildCardType))
        
        type_vars = etype.get_type_variables(self.bt_factory)
        type_params = self.gen_type_params(
            len(type_vars), with_variance=self.language in ['kotlin', 'scala'])
        type_var_map = {}
        available_type_params = list(type_params)
        can_wildcard = False
        for type_var, bounds in type_vars.items():
            # The given type 'etype' has type variables.
            # So, it's not safe to instantiate these type variables with
            # wildcard types. In this way we prevent errors like the following.
            #
            # class A<T> {
            #   B<T> foo();
            # }
            # A<? extends Number> x = new A<>();
            # B<Number> = x.foo(); // error: incompatible types
            # TODO: We may support this case in the future.
            can_wildcard = False
            bounds = list(bounds)
            type_param = ut.random.choice(available_type_params)
            available_type_params.remove(type_param)
            if bounds != [None]:
                type_param.bound = functools.reduce(
                    lambda acc, t: t if t.is_subtype(acc) else acc,
                    filter(lambda t: t is not None, bounds), bounds[0])
            else:
                type_param.bound = None
            type_param.variance = tp.Invariant
            type_var_map[type_var] = type_param
        return type_params, type_var_map, can_wildcard




    def gen_struct_inst(self, struct_decl: ast.StructDeclaration):
        """Initialize a struct with values.
           --deprecated
        """
        prev_move_semantics = self.move_semantics
        self.move_semantics = True
        initial_depth = self.depth
        self.depth += 1
        field_names = [field_decl.name for field_decl in struct_decl.fields]
        field_exprs = []
        for field_decl in struct_decl.fields:
            gen_bottom = (self.depth > (cfg.limits.max_depth * 2))
            matching_expr = self.generate_expr(field_decl.get_type(), only_leaves=True, exclude_var=True, gen_bottom=gen_bottom)
            field_exprs.append(matching_expr)
        self.depth = initial_depth
        self.move_semantics = prev_move_semantics
        return ast.StructInstantiation(struct_decl.name, field_names, field_exprs)

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
        type_params = init_type_params or self.gen_type_params()
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
            structs_that_impl=[],
            type_parameters=type_params,
        )
        self._add_node_to_parent(ast.GLOBAL_NAMESPACE, trait)
        self._blacklisted_traits.add(trait_name)
        supertrait = self._select_supertrait()
        if supertrait:
            trait.supertraits.extend(supertrait)
        self.gen_trait_functions(fret_type=fret_type, not_void=not_void, signature=signature, only_signatures=True)
        #self.gen_func_decl(fret_type, not_void=not_void, is_interface=False, trait_func=True)
        self._inside_trait_decl = False
        self.namespace = initial_namespace
        return trait


    def _select_supertrait(self):
        """Select a supertrait for a trait.
           For now up to one supertrait is selected.
        """
        current_trait = self.namespace[-1]
        trait_decls = [
            t for t in self.context.get_traits(self.namespace).values()
            if t.name != current_trait and t.name not in self._blacklisted_traits
        ]
        if not trait_decls:
            return None
        trait_decl = ut.random.choice(trait_decls)
        if ut.random.bool():
            return [trait_decl]
        else:
            return None

    def gen_trait_functions(self,
                            fret_type=None,
                            not_void=False,
                            signature: tp.ParameterizedType=None,
                            only_signatures=False) -> List[ast.FunctionDeclaration]:
        """ Generate trait signatures and default implementations.
        """
        funcs = []
        max_funcs = cfg.limits.cls.max_funcs - 1 if fret_type else cfg.limits.cls.max_funcs
        if fret_type:
            func = self.gen_func_decl(fret_type, not_void=not_void, is_interface=only_signatures, trait_func=True)
            funcs.append(func)
        if signature:
            ret_type, params = self._gen_ret_and_paramas_from_sig(signature)
            func = self.gen_func_decl(ret_type, params=params, not_void=not_void, is_interface=only_signatures, trait_func=True)
            func.trait_func = True
            funcs.append(func)
        for _ in range(ut.random.integer(0, max_funcs)):
            func = self.gen_func_decl(not_void=not_void, is_interface=only_signatures, trait_func=True)
            funcs.append(func)
        return funcs

    def _update_type_map(self, type_map, impl_type_params, fret_type):
        """ Swap type parameters from Impl block with concrete types
            Args:
                type_map: type map of the struct
                impl_type_params: type parameters of the impl block
                fret_type: return type a function should have
        """

    def gen_matching_impl(self, fret_type: tp.Type) -> gu.AttrAccessInfo:
        if fret_type.is_parameterized():
            trait = self._gen_matching_trait(fret_type, True)
            struct = self.gen_struct_decl(field_type=fret_type)
        impl, struct_decl, type_var_map = self.gen_impl(fret_type)
        return self._get_struct_with_matching_function(fret_type)
 
    def gen_impl(self,
                 fret_type: tp.Type=None,
                 not_void: bool=False,
                 type_params: List[tp.TypeParameter]=None,
                 signature: tp.ParameterizedType=None, #remove this???
                 struct_name: str=None,
                 trait: ast.TraitDeclaration=None,
                 ) -> Tuple[ast.Impl, ast.StructDeclaration, Dict]:
        """Generate an impl block.
           Args:
              fret_type: if provided, at least one function will return this type.
              not_void: do not generate functions that return void.
              type_params: list of type parameters.
              signature: generate at least one function with this signature.
              struct: struct to implement. If None, a random struct is selected.
              trait: trait whose functions are to be implemented. If None and fret_type is None, a random trait is selected.

              Returns:
                
        """
        def _inst_type_constructor(obj):
            ttype, type_var_map = obj.get_type(), {}
            if obj.is_parameterized():
                ttype, type_var_map = self.instantiate_type_constructor(obj.get_type(), self.get_types() + impl_type_params)
            return ttype, type_var_map

        initial_namespace = self.namespace
        self.namespace = ast.GLOBAL_NAMESPACE
        
        #Generate candidate type parameters
        #Type params used in final declaration of impl block will be a subset of these,
        #depending on instantiations of struct and trait
        impl_type_params = self.gen_type_params(add_to_context=False, count=3, for_impl=True)
        
        #find or generate trait
        if fret_type is not None:
            trait = self._get_matching_trait(fret_type)
            if not trait:
                trait = self._gen_matching_trait(fret_type, True)
        elif trait is None:
            available_traits = list(self.context.get_traits(self.namespace).values())
            if available_traits:
                trait = ut.random.choice(available_traits)
            else:
                trait = self.gen_trait_decl()
        #find or generate struct
        if struct_name is None:
            structs_in_context = list(self.context.get_structs(self.namespace).values())
            if not structs_in_context:
                struct = self.gen_struct_decl()
            else:
                struct = ut.random.choice(structs_in_context)
            s_type, type_var_map = _inst_type_constructor(struct)
            if not self._is_impl_allowed(struct, type_var_map, trait):
                struct = self.gen_struct_decl() #if not allowed, just generate a fresh struct
                s_type, type_var_map = _inst_type_constructor(struct)
        else:
            struct = self.gen_struct_decl(struct_name)
            s_type, type_var_map = _inst_type_constructor(struct)
        #Type parameters used in final declaration of impl block are the ones that are also used in struct type instantiation
        s_type_params = s_type.type_args if s_type.is_parameterized() else []
        impl_type_params = [t_param for t_param in impl_type_params if t_param in s_type_params]
        t_type, trait_map = _inst_type_constructor(trait) #Instantiate trait type with available type params
        struct_name = struct.name
        impl_id = self._get_impl_id(str(t_type), str(s_type))
        self.namespace = ast.GLOBAL_NAMESPACE + (impl_id,)

        #Adding impl block info to _impls
        impl = ast.Impl(impl_id, s_type, t_type, [], impl_type_params)
        if not struct_name in self._impls.keys():
            self._impls[struct_name] = []
        self._impls[struct_name].append((impl, type_var_map, trait_map))

        #Adding type parameters to context, so that they can be used in generated functions
        for t_param in impl_type_params:
            self.context.add_type(self.namespace, t_param.name, t_param)
        
        #Adding fields to context so that they are accessible in impl block
        field_vars_list = []
        for field in struct.fields:
            field_type = tp.substitute_type(field.get_type(), type_var_map)
            field_var_name ="self." + field.name
            #create a virtual variable declaration for each struct field
            var_decl = ast.VariableDeclaration(name=field_var_name, expr=None, var_type=field_type)
            var_decl.move_prohibited = self._type_moveable(var_decl)
            self.context.add_var(self.namespace, field_var_name, var_decl)
        
        functions = []
        for func_decl in trait.function_signatures:
            prev_ns = self.namespace
            self.namespace += (func_decl.name,)
            new_func = self._update_func_decl(func_decl, trait_map)
            new_func.body = self._gen_func_body(new_func.get_type())
            functions.append(new_func)
            self.namespace = prev_ns
        for func_decl in trait.default_impls:
            prev_ns = self.namespace
            self.namespace += (func_decl.name,)
            new_func = self._update_func_decl(func_decl, trait_map)
            if ut.random.bool():
                #decide randomly to override the default implementation
                new_func.body = self._gen_func_body(new_func.get_type())
                functions.append(new_func)
            self.namespace = prev_ns

        trait.structs_that_impl.append(struct)
        impl.functions = functions
        
        msg = ("Creating impl {}").format(impl_id)
        log(self.logger, msg)

        self._add_node_to_parent(ast.GLOBAL_NAMESPACE, impl)
        self.namespace = initial_namespace
        
        return (impl, struct, type_var_map)


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
        trait_decls = self._get_matching_trait_decls(fret_type, False)
        if not trait_decls:
            return None
        t, type_var_map, func = ut.random.choice(trait_decls)
        return t

    def _get_matching_trait_decls(self, 
                                  fret_type: tp.Type,
                                  subtype: bool) -> List[Tuple[ast.TraitDeclaration, tu.TypeVarMap, ast.FunctionDeclaration]]:
        """ Get traits that have functions that return fret_type.
        """
        trait_decls = []
        for t in self.context.get_traits(self.namespace).values():
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