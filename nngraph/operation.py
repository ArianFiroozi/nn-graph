import torch
from graphviz import Digraph
import pickle
import torch.nn as nn
from enum import Enum

class OperationType(Enum):
    INPUT=1
    MAC=2
    ADD=3
    MULT=4
    OUTPUT=5
    POOL=6
    PADD=7
    DILATION=8
    CONST=9
    PROJECT=10
    DOT_PRODUCT=11
    UNKNOWN=0

class Operation:
    def __init__(self, name:str, type:OperationType=0, label:str="OP"):
        self.name=name # unique
        self.label=label
        self.type=type
        self.inputs=[]
    
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Operation) and self.name == other.name

    def __repr__(self):
        return self.label

    def get_label(self):
        return self.label

    def add_input_name(self, name:str):
        self.inputs.append(name)
    
    def get_name(self)->str:
        return self.name

class PaddOP(Operation):
    def __init__(self, name:str, padding:list, label:str="Padding"):
        super().__init__(name, OperationType.PADD, label)
        self.padding = padding

    def get_label(self):
        return self.label + ": " + str(self.padding)

class DilationOP(Operation):
    def __init__(self, name:str, dilation:list, label:str="Dilation"):
        super().__init__(name, OperationType.DILATION, label)
        self.dilation = dilation

    def get_label(self):
        return self.label + ": " + str(self.dilation)

class MacOP(Operation):
    def __init__(self, name:str, input_index, weight_index, weight=None, label:str="MAC"):
        super().__init__(name, OperationType.MAC, label)
        self.input_index=input_index
        self.weight_index=weight_index
        self.weight=weight

    def get_label(self):
        printable = self.label + ":" 
        printable += "\nX" + str(self.input_index) 
        printable += "\nW" + str(self.weight_index)

        if isinstance(self.weight, torch.Tensor):
            printable += "\nWeight: " + str([i.item() for i in self.weight])

        return printable

class ConstOP(Operation):
    def __init__(self, name:str, shape:list, label:str="Const"):
        super().__init__(name, OperationType.CONST, label)
        self.shape = shape

    def get_label(self):
        printable = self.label + ":" 
        printable += "\nW" + str(self.shape)
        return printable

class InputOP(Operation):
    def __init__(self, name:str, shape:list, label:str="Input"):
        super().__init__(name, OperationType.INPUT, label)
        self.shape = shape

    def get_label(self):
        printable = self.label + ":" 
        printable += "\nX" + str(self.shape)
        return printable

class OutputOP(Operation):
    def __init__(self, name:str, shape:list, label:str="Output"):
        super().__init__(name, OperationType.OUTPUT, label)
        self.shape = shape

    def get_label(self):
        printable = self.label + ":" 
        printable += "\nY" + str(self.shape)
        return printable

class ProjectOP(Operation):
    def __init__(self, name:str, shape:list, label:str="Project"):
        super().__init__(name, OperationType.PROJECT, label)
        self.shape = shape

    def get_label(self):
        printable = self.label + ":" 
        printable += "\nW" + str(self.shape)
        return printable

class DotProduct(Operation):
    def __init__(self, name:str, input1_index, input2_index, label:str="Dot Product"):
        super().__init__(name, OperationType.DOT_PRODUCT, label)
        self.input1_index=input1_index
        self.input2_index=input2_index

    def get_label(self):
        printable = self.label + ":" 
        printable += "\ninp1" + str(self.input1_index) 
        printable += "\ninp2" + str(self.input2_index)
        return printable