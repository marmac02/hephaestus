from src.ir.types import Builtin

import src.ir.builtins as bt
import src.ir.types as tp
from typing import List

class RustBuiltinFactory(bt.BuiltinFactory):
    def get_language(self):
        return "rust"

    def get_builtin(self):
        return RustBuiltin

    def get_void_type(self):
        return UnitType()
    
    def get_number_type(self):
        return I32()

    def get_integer_type(self, primitive=False):
        return I32()

    def get_byte_type(self, primitive=False):
        return I8()

    def get_short_type(self, primitive=False):
        return I16()

    def get_long_type(self, primitive=False):
        return I64()

    def get_float_type(self, primitive=False):
        return F32()

    def get_double_type(self, primitive=False):
        return F64()

    def get_big_decimal_type(self):
        return F64()

    def get_big_integer_type(self):
        return I64()

    def get_boolean_type(self, primitive=False):
        return BoolType()

    def get_char_type(self, primitive=False):
        return CharType()

    def get_string_type(self):
        return StringType()

    def get_array_type(self): 
        return VecType()

    def get_function_type(self, nr_type_parameters=0):
        return FunctionType(nr_type_parameters)

    def get_any_type(self):
        return AnyType()

    def get_non_nothing_types(self):
        types = super().get_non_nothing_types()
        if AnyType() in types:
            types.remove(AnyType())
        return types
    
    def get_Fn_type(self, nr_type_parameters=0):
        return Fn(nr_type_parameters)
    
    def get_FnMut_type(self, nr_type_parameters=0):
        return FnMut(nr_type_parameters)
    
    def get_FnOnce_type(self, nr_type_parameters=0):
        return FnOnce(nr_type_parameters)

    def get_function_trait_types(self, max_parameters):
        return [self.get_Fn_type(i) for i in range(0, max_parameters + 1)] + \
               [self.get_FnMut_type(i) for i in range(0, max_parameters + 1)] + \
               [self.get_FnOnce_type(i) for i in range(0, max_parameters + 1)]

def is_function_constructor(t: tp.Type):
    return t.name == "fn"

def is_function_trait(t: tp.Type):
    return getattr(t, "is_parameterized", lambda: False)() and \
        getattr(t.t_constructor, "is_function_trait", lambda: False)()

def is_Fn(t: tp.Type):
    return t.is_parameterized() and \
        isinstance(t.t_constructor, Fn)

def is_FnMut(t: tp.Type):
    return t.is_parameterized() and \
        isinstance(t.t_constructor, FnMut)

def is_FnOnce(t: tp.Type):
    return t.is_parameterized() and \
        isinstance(t.t_constructor, FnOnce)


class RustBuiltin(tp.Builtin):

    def __init__(self, name, primitive=False):
        super().__init__(name)
        self.primitive = primitive

    def __str__(self):
        return str(self.name) + "(rust-builtin)"
    
    def is_primitive(self):
        return self.primitive

class UnitType(RustBuiltin):
    def __init__(self, name="()"):
        super().__init__(name)

    def get_builtin_type(self):
        return bt.Void

class I8(RustBuiltin):
    def __init__(self, name="i8"):
        super().__init__(name, True)

    def get_builtin_type(self):
        return bt.Byte

class I16(RustBuiltin):
    def __init__(self, name="i16"):
        super().__init__(name, True)

class I32(RustBuiltin):
    def __init__(self, name="i32"):
        super().__init__(name, True)
    def get_builtin_type(self):
        return bt.Int

class I64(RustBuiltin):
    def __init__(self, name="i64"):
        super().__init__(name, True)
    def get_builtin_type(self):
        return bt.Long

class I128(RustBuiltin):
    def __init__(self, name="i128"):
        super().__init__(name, True)

class U8(RustBuiltin):
    def __init__(self, name="u8"):
        super().__init__(name, True)

class U16(RustBuiltin):
    def __init__(self, name="u16"):
        super().__init__(name, True)

class U32(RustBuiltin):
    def __init__(self, name="u32"):
        super().__init__(name, True)

class U64(RustBuiltin):
    def __init__(self, name="u64"):
        super().__init__(name, True)

class U128(RustBuiltin):
    def __init__(self, name="u128"):
        super().__init__(name, True)

class F32(RustBuiltin):
    def __init__(self, name="f32"):
        super().__init__(name, True)
    def get_builtin_type(self):
        return bt.Float

class F64(RustBuiltin):
    def __init__(self, name="f64"):
        super().__init__(name, True)
    def get_builtin_type(self):
        return bt.Double

class CharType(RustBuiltin):
    def __init__(self, name="char"):
        super().__init__(name, True)
    def get_builtin_type(self):
        return bt.Char

class StringType(RustBuiltin):
    def __init__(self, name="String"):
        super().__init__(name)
    def get_builtin_type(self):
        return bt.String

class BoolType(RustBuiltin):
    def __init__(self, name="bool"):
        super().__init__(name, True)
    def get_builtin_type(self):
        return bt.Boolean

class VecType(tp.TypeConstructor, RustBuiltin):
    def __init__(self, name="Vec"):
        super().__init__(name, [tp.TypeParameter("T")])
        super(RustBuiltin, self).__init__(name)

class FunctionType(tp.TypeConstructor):
    def __init__(self, nr_type_parameters: int, name="fn"):
        fn_name = name
        type_parameters = [tp.TypeParameter("A" + str(i), tp.Contravariant)
            for i in range(1, nr_type_parameters + 1)] + [tp.TypeParameter("R", tp.Covariant)]
        self.nr_type_parameters = nr_type_parameters
        super().__init__(fn_name, type_parameters)

class Fn(FunctionType):
    name = "Fn"
    def __init__(self, nr_type_parameters: int):
        super().__init__(nr_type_parameters, name="Fn")
    
    def is_function_trait(self):
        return True

class FnMut(FunctionType):
    def __init__(self, nr_type_parameters: int):
        super().__init__(nr_type_parameters, name="FnMut")
   
    def is_function_trait(self):
        return True

class FnOnce(FunctionType):
    def __init__(self, nr_type_parameters: int):
        super().__init__(nr_type_parameters, name="FnOnce")
    
    def is_function_trait(self):
        return True

class AnyType(RustBuiltin):
    def __init__(self, name="ANY"):
        super().__init__(name)
    def get_builtin_type(self):
        return bt.Any

class AnyRefType(AnyType):
    def __init__(self, name="AnyRef"):
        super().__init__(name)


Unit = UnitType()