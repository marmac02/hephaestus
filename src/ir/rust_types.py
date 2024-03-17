from src.ir.types import Builtin

import src.ir.builtins as bt
import src.ir.types as tp

#tempoprary types to fit with generator

class RustBuiltinFactory(bt.BuiltinFactory):
    def get_language(self):
        return "rust"

    def get_builtin(self):
        return RustBuiltin

    def get_unit_type(self):
        return UnitType()

    #AnyType check if relevant (dyn Any trait)

    def get_number_type(self): #check if relevant
        return NumberType()

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
        return ArrayType()


class RustBuiltin(tp.Builtin):
    def __str__(self):
        return str(self.name) + "(rust-builtin)"
    
    def is_primitive(self):
        return False

class UnitType(RustBuiltin):
    def __init__(self, name="()"):
        super().__init__(name)

    def get_builtin_type(self):
        return bt.Void

class I8(RustBuiltin):
    def __init__(self, name="i8"):
        super().__init__(name)

    def get_builtin_type(self):
        return bt.Byte

class I16(RustBuiltin):
    def __init__(self, name="i16"):
        super().__init__(name)
    #handle buildin type

class I32(RustBuiltin):
    def __init__(self, name="i32"):
        super().__init__(name)
    #handle buildin type
    def get_builtin_type(self):
        return bt.Int

class I64(RustBuiltin):
    def __init__(self, name="i64"):
        super().__init__(name)
    #handle buildin type
    def get_builtin_type(self):
        return bt.Long

class I128(RustBuiltin):
    def __init__(self, name="i128"):
        super().__init__(name)
    #handle buildin type

class U8(RustBuiltin):
    def __init__(self, name="u8"):
        super().__init__(name)
    #handle buildin type

class U16(RustBuiltin):
    def __init__(self, name="u16"):
        super().__init__(name)
    #handle buildin type

class U32(RustBuiltin):
    def __init__(self, name="u32"):
        super().__init__(name)
    #handle buildin type

class U64(RustBuiltin):
    def __init__(self, name="u64"):
        super().__init__(name)
    #handle buildin type

class U128(RustBuiltin):
    def __init__(self, name="u128"):
        super().__init__(name)
    #handle buildin type

class F32(RustBuiltin):
    def __init__(self, name="f32"):
        super().__init__(name)
    #handle buildin type
    def get_builtin_type(self):
        return bt.Float

class F64(RustBuiltin):
    def __init__(self, name="f64"):
        super().__init__(name)
    #handle buildin type
    def get_builtin_type(self):
        return bt.Double

class CharType(RustBuiltin):
    def __init__(self, name="char"):
        super().__init__(name)
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
        super().__init__(name)
    #handle buildin type
    def get_builtin_type(self):
        return bt.Boolean

class ArrayType(tp.TypeConstructor, RustBuiltin):
    pass

class FunctionType(tp.TypeConstructor, RustBuiltin):
    pass

class TupleType(tp.TypeConstructor, RustBuiltin):
    pass