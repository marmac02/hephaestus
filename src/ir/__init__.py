from src.ir.kotlin_types import KotlinBuiltinFactory
from src.ir.groovy_types import GroovyBuiltinFactory
from src.ir.java_types import JavaBuiltinFactory
from src.ir.scala_types import ScalaBuiltinFactory
from src.ir.rust_types import RustBuiltinFactory


BUILTIN_FACTORIES = {
    "kotlin": KotlinBuiltinFactory(),
    "groovy": GroovyBuiltinFactory(),
    "java": JavaBuiltinFactory(),
    "scala": ScalaBuiltinFactory(),
    "rust": RustBuiltinFactory()
}
