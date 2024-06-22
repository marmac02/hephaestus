from src.ir.types import Builtin

import src.ir.builtins as bt
import src.ir.types as tp
from typing import List

#tempoprary types to fit with generator

class RustBuiltinFactory(bt.BuiltinFactory):
    def get_language(self):
        return "rust"

    def get_builtin(self):
        return RustBuiltin

    def get_void_type(self):
        return UnitType()
    
    #this must be fixed
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
            types.remove(AnyType()) #might change later
        return types
    
    def get_fn_trait_classes(self):
        return [Fn]


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
        #self.supertypes.append(AnyType())

    def get_builtin_type(self):
        return bt.Void

class I8(RustBuiltin):
    def __init__(self, name="i8"):
        super().__init__(name, True)
        #self.supertypes.append(AnyType())

    def get_builtin_type(self):
        return bt.Byte

class I16(RustBuiltin):
    def __init__(self, name="i16"):
        super().__init__(name, True)
        #self.supertypes.append(AnyType())
    #handle buildin type

class I32(RustBuiltin):
    def __init__(self, name="i32"):
        super().__init__(name, True)
        #self.supertypes.append(AnyType())
    #handle buildin type
    def get_builtin_type(self):
        return bt.Int

class I64(RustBuiltin):
    def __init__(self, name="i64"):
        super().__init__(name, True)
        #self.supertypes.append(AnyType())
    #handle buildin type
    def get_builtin_type(self):
        return bt.Long

class I128(RustBuiltin):
    def __init__(self, name="i128"):
        super().__init__(name, True)
        #self.supertypes.append(AnyType())
    #handle buildin type

class U8(RustBuiltin):
    def __init__(self, name="u8"):
        super().__init__(name, True)
        #self.supertypes.append(AnyType())
    #handle buildin type

class U16(RustBuiltin):
    def __init__(self, name="u16"):
        super().__init__(name, True)
        #self.supertypes.append(AnyType())
    #handle buildin type

class U32(RustBuiltin):
    def __init__(self, name="u32"):
        super().__init__(name, True)
        #self.supertypes.append(AnyType())
    #handle buildin type

class U64(RustBuiltin):
    def __init__(self, name="u64"):
        super().__init__(name, True)
        #self.supertypes.append(AnyType())
    #handle buildin type

class U128(RustBuiltin):
    def __init__(self, name="u128"):
        super().__init__(name, True)
        #self.supertypes.append(AnyType())
    #handle buildin type

class F32(RustBuiltin):
    def __init__(self, name="f32"):
        super().__init__(name, True)
        #self.supertypes.append(AnyType())
    #handle buildin type
    def get_builtin_type(self):
        return bt.Float

class F64(RustBuiltin):
    def __init__(self, name="f64"):
        super().__init__(name, True)
        #self.supertypes.append(AnyType())
    #handle buildin type
    def get_builtin_type(self):
        return bt.Double

class CharType(RustBuiltin):
    def __init__(self, name="char"):
        super().__init__(name, True)
        #self.supertypes.append(AnyType())
    #handle buildin type
    def get_builtin_type(self):
        return bt.Char

#for now handling only String and not &str
class StringType(RustBuiltin):
    def __init__(self, name="String"):
        super().__init__(name)
    #handle buildin type
    def get_builtin_type(self):
        return bt.String

class BoolType(RustBuiltin):
    def __init__(self, name="bool"):
        super().__init__(name, True)
        #self.supertypes.append(AnyType())
    #handle buildin type
    def get_builtin_type(self):
        return bt.Boolean

#change this, for now it is Vec<T>
class VecType(tp.TypeConstructor, RustBuiltin):
    def __init__(self, name="Vec"):
        super().__init__(name, [tp.TypeParameter("T")])
        super(RustBuiltin, self).__init__(name)
        #self.supertypes.append(AnyRefType())

#fix covariance/contra-variance
class FunctionType(tp.TypeConstructor):
    def __init__(self, nr_type_parameters: int, name="fn"):
        fn_name = name
        type_parameters = [tp.TypeParameter("A" + str(i), tp.Contravariant)  #was tp.Contravariant
            for i in range(1, nr_type_parameters + 1)] + [tp.TypeParameter("R", tp.Covariant)] #was 
        self.nr_type_parameters = nr_type_parameters
        super().__init__(fn_name, type_parameters)


class Fn(FunctionType):
    name = "Fn"
    def __init__(self, nr_type_parameters: int):
        super().__init__(nr_type_parameters, name="Fn")

class FnMut(FunctionType):
    def __init__(self, nr_type_parameters: int):
        super().__init__(nr_type_parameters, name="FnMut")

class FnOnce(FunctionType):
    def __init__(self, nr_type_parameters: int):
        super().__init__(nr_type_parameters, name="FnOnce")

#erase AnyType (not relevant to rust)
class AnyType(RustBuiltin):
    def __init__(self, name="ANY"): #change this definitely
        super().__init__(name)
    #handle buildin type
    def get_builtin_type(self):
        return bt.Any

class AnyRefType(AnyType):
    def __init__(self, name="AnyRef"):
        super().__init__(name)
    #handle buildin type
        #self.supertypes.append(AnyType())

class TupleType(tp.TypeConstructor, RustBuiltin):
    pass


Unit = UnitType()