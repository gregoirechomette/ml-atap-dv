??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
v
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	@*
shared_namedense1/kernel
o
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel*
_output_shapes

:	@*
dtype0
n
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense1/bias
g
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes
:@*
dtype0
w
dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*
shared_namedense2/kernel
p
!dense2/kernel/Read/ReadVariableOpReadVariableOpdense2/kernel*
_output_shapes
:	@?*
dtype0
o
dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense2/bias
h
dense2/bias/Read/ReadVariableOpReadVariableOpdense2/bias*
_output_shapes	
:?*
dtype0
x
dense3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense3/kernel
q
!dense3/kernel/Read/ReadVariableOpReadVariableOpdense3/kernel* 
_output_shapes
:
??*
dtype0
o
dense3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense3/bias
h
dense3/bias/Read/ReadVariableOpReadVariableOpdense3/bias*
_output_shapes	
:?*
dtype0
w
dense4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense4/kernel
p
!dense4/kernel/Read/ReadVariableOpReadVariableOpdense4/kernel*
_output_shapes
:	?*
dtype0
n
dense4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense4/bias
g
dense4/bias/Read/ReadVariableOpReadVariableOpdense4/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	@*%
shared_nameAdam/dense1/kernel/m
}
(Adam/dense1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/m*
_output_shapes

:	@*
dtype0
|
Adam/dense1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/dense1/bias/m
u
&Adam/dense1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*%
shared_nameAdam/dense2/kernel/m
~
(Adam/dense2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense2/kernel/m*
_output_shapes
:	@?*
dtype0
}
Adam/dense2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_nameAdam/dense2/bias/m
v
&Adam/dense2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*%
shared_nameAdam/dense3/kernel/m

(Adam/dense3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense3/kernel/m* 
_output_shapes
:
??*
dtype0
}
Adam/dense3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_nameAdam/dense3/bias/m
v
&Adam/dense3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense3/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/dense4/kernel/m
~
(Adam/dense4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense4/kernel/m*
_output_shapes
:	?*
dtype0
|
Adam/dense4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/dense4/bias/m
u
&Adam/dense4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense4/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	@*%
shared_nameAdam/dense1/kernel/v
}
(Adam/dense1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/v*
_output_shapes

:	@*
dtype0
|
Adam/dense1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/dense1/bias/v
u
&Adam/dense1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*%
shared_nameAdam/dense2/kernel/v
~
(Adam/dense2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense2/kernel/v*
_output_shapes
:	@?*
dtype0
}
Adam/dense2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_nameAdam/dense2/bias/v
v
&Adam/dense2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*%
shared_nameAdam/dense3/kernel/v

(Adam/dense3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense3/kernel/v* 
_output_shapes
:
??*
dtype0
}
Adam/dense3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_nameAdam/dense3/bias/v
v
&Adam/dense3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense3/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/dense4/kernel/v
~
(Adam/dense4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense4/kernel/v*
_output_shapes
:	?*
dtype0
|
Adam/dense4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/dense4/bias/v
u
&Adam/dense4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?-
value?-B?- B?-
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
?
$iter

%beta_1

&beta_2
	'decay
(learning_ratemMmNmOmPmQmRmSmTvUvVvWvXvYvZv[v\
 
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
?
regularization_losses
	variables
)non_trainable_variables
	trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
 
YW
VARIABLE_VALUEdense1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
	variables
.non_trainable_variables
trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
YW
VARIABLE_VALUEdense2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
	variables
3non_trainable_variables
trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
YW
VARIABLE_VALUEdense3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
	variables
8non_trainable_variables
trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
YW
VARIABLE_VALUEdense4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
 regularization_losses
!	variables
=non_trainable_variables
"trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4

B0
C1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Dtotal
	Ecount
F	variables
G	keras_api
D
	Htotal
	Icount
J
_fn_kwargs
K	variables
L	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

D0
E1

F	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

H0
I1

K	variables
|z
VARIABLE_VALUEAdam/dense1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_input1Placeholder*'
_output_shapes
:?????????	*
dtype0*
shape:?????????	
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input1dense1/kerneldense1/biasdense2/kerneldense2/biasdense3/kerneldense3/biasdense4/kerneldense4/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_4590
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp!dense2/kernel/Read/ReadVariableOpdense2/bias/Read/ReadVariableOp!dense3/kernel/Read/ReadVariableOpdense3/bias/Read/ReadVariableOp!dense4/kernel/Read/ReadVariableOpdense4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/dense1/kernel/m/Read/ReadVariableOp&Adam/dense1/bias/m/Read/ReadVariableOp(Adam/dense2/kernel/m/Read/ReadVariableOp&Adam/dense2/bias/m/Read/ReadVariableOp(Adam/dense3/kernel/m/Read/ReadVariableOp&Adam/dense3/bias/m/Read/ReadVariableOp(Adam/dense4/kernel/m/Read/ReadVariableOp&Adam/dense4/bias/m/Read/ReadVariableOp(Adam/dense1/kernel/v/Read/ReadVariableOp&Adam/dense1/bias/v/Read/ReadVariableOp(Adam/dense2/kernel/v/Read/ReadVariableOp&Adam/dense2/bias/v/Read/ReadVariableOp(Adam/dense3/kernel/v/Read/ReadVariableOp&Adam/dense3/bias/v/Read/ReadVariableOp(Adam/dense4/kernel/v/Read/ReadVariableOp&Adam/dense4/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_5003
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense1/kerneldense1/biasdense2/kerneldense2/biasdense3/kerneldense3/biasdense4/kerneldense4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense1/kernel/mAdam/dense1/bias/mAdam/dense2/kernel/mAdam/dense2/bias/mAdam/dense3/kernel/mAdam/dense3/bias/mAdam/dense4/kernel/mAdam/dense4/bias/mAdam/dense1/kernel/vAdam/dense1/bias/vAdam/dense2/kernel/vAdam/dense2/bias/vAdam/dense3/kernel/vAdam/dense3/bias/vAdam/dense4/kernel/vAdam/dense4/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_5112??
?	
?
$__inference_model_layer_call_fn_4314

input1
unknown:	@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_42952
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinput1
?+
?
__inference__wrapped_model_4183

input1=
+model_dense1_matmul_readvariableop_resource:	@:
,model_dense1_biasadd_readvariableop_resource:@>
+model_dense2_matmul_readvariableop_resource:	@?;
,model_dense2_biasadd_readvariableop_resource:	??
+model_dense3_matmul_readvariableop_resource:
??;
,model_dense3_biasadd_readvariableop_resource:	?>
+model_dense4_matmul_readvariableop_resource:	?:
,model_dense4_biasadd_readvariableop_resource:
identity??#model/dense1/BiasAdd/ReadVariableOp?"model/dense1/MatMul/ReadVariableOp?#model/dense2/BiasAdd/ReadVariableOp?"model/dense2/MatMul/ReadVariableOp?#model/dense3/BiasAdd/ReadVariableOp?"model/dense3/MatMul/ReadVariableOp?#model/dense4/BiasAdd/ReadVariableOp?"model/dense4/MatMul/ReadVariableOp?
"model/dense1/MatMul/ReadVariableOpReadVariableOp+model_dense1_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype02$
"model/dense1/MatMul/ReadVariableOp?
model/dense1/MatMulMatMulinput1*model/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/dense1/MatMul?
#model/dense1/BiasAdd/ReadVariableOpReadVariableOp,model_dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#model/dense1/BiasAdd/ReadVariableOp?
model/dense1/BiasAddBiasAddmodel/dense1/MatMul:product:0+model/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/dense1/BiasAdd
model/dense1/ReluRelumodel/dense1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model/dense1/Relu?
"model/dense2/MatMul/ReadVariableOpReadVariableOp+model_dense2_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02$
"model/dense2/MatMul/ReadVariableOp?
model/dense2/MatMulMatMulmodel/dense1/Relu:activations:0*model/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense2/MatMul?
#model/dense2/BiasAdd/ReadVariableOpReadVariableOp,model_dense2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#model/dense2/BiasAdd/ReadVariableOp?
model/dense2/BiasAddBiasAddmodel/dense2/MatMul:product:0+model/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense2/BiasAdd?
model/dense2/ReluRelumodel/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/dense2/Relu?
"model/dense3/MatMul/ReadVariableOpReadVariableOp+model_dense3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"model/dense3/MatMul/ReadVariableOp?
model/dense3/MatMulMatMulmodel/dense2/Relu:activations:0*model/dense3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense3/MatMul?
#model/dense3/BiasAdd/ReadVariableOpReadVariableOp,model_dense3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#model/dense3/BiasAdd/ReadVariableOp?
model/dense3/BiasAddBiasAddmodel/dense3/MatMul:product:0+model/dense3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense3/BiasAdd?
model/dense3/ReluRelumodel/dense3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/dense3/Relu?
"model/dense4/MatMul/ReadVariableOpReadVariableOp+model_dense4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"model/dense4/MatMul/ReadVariableOp?
model/dense4/MatMulMatMulmodel/dense3/Relu:activations:0*model/dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense4/MatMul?
#model/dense4/BiasAdd/ReadVariableOpReadVariableOp,model_dense4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/dense4/BiasAdd/ReadVariableOp?
model/dense4/BiasAddBiasAddmodel/dense4/MatMul:product:0+model/dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense4/BiasAdd?
model/dense4/SigmoidSigmoidmodel/dense4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/dense4/Sigmoids
IdentityIdentitymodel/dense4/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp$^model/dense1/BiasAdd/ReadVariableOp#^model/dense1/MatMul/ReadVariableOp$^model/dense2/BiasAdd/ReadVariableOp#^model/dense2/MatMul/ReadVariableOp$^model/dense3/BiasAdd/ReadVariableOp#^model/dense3/MatMul/ReadVariableOp$^model/dense4/BiasAdd/ReadVariableOp#^model/dense4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????	: : : : : : : : 2J
#model/dense1/BiasAdd/ReadVariableOp#model/dense1/BiasAdd/ReadVariableOp2H
"model/dense1/MatMul/ReadVariableOp"model/dense1/MatMul/ReadVariableOp2J
#model/dense2/BiasAdd/ReadVariableOp#model/dense2/BiasAdd/ReadVariableOp2H
"model/dense2/MatMul/ReadVariableOp"model/dense2/MatMul/ReadVariableOp2J
#model/dense3/BiasAdd/ReadVariableOp#model/dense3/BiasAdd/ReadVariableOp2H
"model/dense3/MatMul/ReadVariableOp"model/dense3/MatMul/ReadVariableOp2J
#model/dense4/BiasAdd/ReadVariableOp#model/dense4/BiasAdd/ReadVariableOp2H
"model/dense4/MatMul/ReadVariableOp"model/dense4/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinput1
?	
?
"__inference_signature_wrapper_4590

input1
unknown:	@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_41832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinput1
?
?
@__inference_dense1_layer_call_and_return_conditional_losses_4207

inputs0
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?/dense1/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp?
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square?
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const?
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum?
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x?
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense1/kernel/Regularizer/Square/ReadVariableOp/dense1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?3
?
?__inference_model_layer_call_and_return_conditional_losses_4419

inputs
dense1_4380:	@
dense1_4382:@
dense2_4385:	@?
dense2_4387:	?
dense3_4390:
??
dense3_4392:	?
dense4_4395:	?
dense4_4397:
identity??dense1/StatefulPartitionedCall?/dense1/kernel/Regularizer/Square/ReadVariableOp?dense2/StatefulPartitionedCall?/dense2/kernel/Regularizer/Square/ReadVariableOp?dense3/StatefulPartitionedCall?/dense3/kernel/Regularizer/Square/ReadVariableOp?dense4/StatefulPartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_4380dense1_4382*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_42072 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_4385dense2_4387*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_42302 
dense2/StatefulPartitionedCall?
dense3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0dense3_4390dense3_4392*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense3_layer_call_and_return_conditional_losses_42532 
dense3/StatefulPartitionedCall?
dense4/StatefulPartitionedCallStatefulPartitionedCall'dense3/StatefulPartitionedCall:output:0dense4_4395dense4_4397*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense4_layer_call_and_return_conditional_losses_42702 
dense4/StatefulPartitionedCall?
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense1_4380*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp?
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square?
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const?
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum?
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x?
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mul?
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense2_4385*
_output_shapes
:	@?*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp?
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2"
 dense2/kernel/Regularizer/Square?
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const?
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum?
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x?
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/mul?
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense3_4390* 
_output_shapes
:
??*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp?
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2"
 dense3/kernel/Regularizer/Square?
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const?
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum?
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x?
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/mul?
IdentityIdentity'dense4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense1/StatefulPartitionedCall0^dense1/kernel/Regularizer/Square/ReadVariableOp^dense2/StatefulPartitionedCall0^dense2/kernel/Regularizer/Square/ReadVariableOp^dense3/StatefulPartitionedCall0^dense3/kernel/Regularizer/Square/ReadVariableOp^dense4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????	: : : : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2b
/dense1/kernel/Regularizer/Square/ReadVariableOp/dense1/kernel/Regularizer/Square/ReadVariableOp2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2b
/dense2/kernel/Regularizer/Square/ReadVariableOp/dense2/kernel/Regularizer/Square/ReadVariableOp2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2b
/dense3/kernel/Regularizer/Square/ReadVariableOp/dense3/kernel/Regularizer/Square/ReadVariableOp2@
dense4/StatefulPartitionedCalldense4/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
@__inference_dense4_layer_call_and_return_conditional_losses_4839

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_dense3_layer_call_fn_4828

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense3_layer_call_and_return_conditional_losses_42532
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_dense4_layer_call_and_return_conditional_losses_4270

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_dense3_layer_call_and_return_conditional_losses_4253

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?/dense3/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp?
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2"
 dense3/kernel/Regularizer/Square?
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const?
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum?
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x?
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/muln
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense3/kernel/Regularizer/Square/ReadVariableOp/dense3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_4870K
8dense2_kernel_regularizer_square_readvariableop_resource:	@?
identity??/dense2/kernel/Regularizer/Square/ReadVariableOp?
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense2_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	@?*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp?
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2"
 dense2/kernel/Regularizer/Square?
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const?
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum?
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x?
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/mulk
IdentityIdentity!dense2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp0^dense2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense2/kernel/Regularizer/Square/ReadVariableOp/dense2/kernel/Regularizer/Square/ReadVariableOp
?G
?
__inference__traced_save_5003
file_prefix,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop,
(savev2_dense2_kernel_read_readvariableop*
&savev2_dense2_bias_read_readvariableop,
(savev2_dense3_kernel_read_readvariableop*
&savev2_dense3_bias_read_readvariableop,
(savev2_dense4_kernel_read_readvariableop*
&savev2_dense4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_dense1_kernel_m_read_readvariableop1
-savev2_adam_dense1_bias_m_read_readvariableop3
/savev2_adam_dense2_kernel_m_read_readvariableop1
-savev2_adam_dense2_bias_m_read_readvariableop3
/savev2_adam_dense3_kernel_m_read_readvariableop1
-savev2_adam_dense3_bias_m_read_readvariableop3
/savev2_adam_dense4_kernel_m_read_readvariableop1
-savev2_adam_dense4_bias_m_read_readvariableop3
/savev2_adam_dense1_kernel_v_read_readvariableop1
-savev2_adam_dense1_bias_v_read_readvariableop3
/savev2_adam_dense2_kernel_v_read_readvariableop1
-savev2_adam_dense2_bias_v_read_readvariableop3
/savev2_adam_dense3_kernel_v_read_readvariableop1
-savev2_adam_dense3_bias_v_read_readvariableop3
/savev2_adam_dense4_kernel_v_read_readvariableop1
-savev2_adam_dense4_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableop(savev2_dense3_kernel_read_readvariableop&savev2_dense3_bias_read_readvariableop(savev2_dense4_kernel_read_readvariableop&savev2_dense4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_dense1_kernel_m_read_readvariableop-savev2_adam_dense1_bias_m_read_readvariableop/savev2_adam_dense2_kernel_m_read_readvariableop-savev2_adam_dense2_bias_m_read_readvariableop/savev2_adam_dense3_kernel_m_read_readvariableop-savev2_adam_dense3_bias_m_read_readvariableop/savev2_adam_dense4_kernel_m_read_readvariableop-savev2_adam_dense4_bias_m_read_readvariableop/savev2_adam_dense1_kernel_v_read_readvariableop-savev2_adam_dense1_bias_v_read_readvariableop/savev2_adam_dense2_kernel_v_read_readvariableop-savev2_adam_dense2_bias_v_read_readvariableop/savev2_adam_dense3_kernel_v_read_readvariableop-savev2_adam_dense3_bias_v_read_readvariableop/savev2_adam_dense4_kernel_v_read_readvariableop-savev2_adam_dense4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	@:@:	@?:?:
??:?:	?:: : : : : : : : : :	@:@:	@?:?:
??:?:	?::	@:@:	@?:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:	@: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:	@: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:	@: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:% !

_output_shapes
:	?: !

_output_shapes
::"

_output_shapes
: 
?
?
__inference_loss_fn_2_4881L
8dense3_kernel_regularizer_square_readvariableop_resource:
??
identity??/dense3/kernel/Regularizer/Square/ReadVariableOp?
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense3_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp?
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2"
 dense3/kernel/Regularizer/Square?
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const?
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum?
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x?
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/mulk
IdentityIdentity!dense3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp0^dense3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense3/kernel/Regularizer/Square/ReadVariableOp/dense3/kernel/Regularizer/Square/ReadVariableOp
?D
?
?__inference_model_layer_call_and_return_conditional_losses_4690

inputs7
%dense1_matmul_readvariableop_resource:	@4
&dense1_biasadd_readvariableop_resource:@8
%dense2_matmul_readvariableop_resource:	@?5
&dense2_biasadd_readvariableop_resource:	?9
%dense3_matmul_readvariableop_resource:
??5
&dense3_biasadd_readvariableop_resource:	?8
%dense4_matmul_readvariableop_resource:	?4
&dense4_biasadd_readvariableop_resource:
identity??dense1/BiasAdd/ReadVariableOp?dense1/MatMul/ReadVariableOp?/dense1/kernel/Regularizer/Square/ReadVariableOp?dense2/BiasAdd/ReadVariableOp?dense2/MatMul/ReadVariableOp?/dense2/kernel/Regularizer/Square/ReadVariableOp?dense3/BiasAdd/ReadVariableOp?dense3/MatMul/ReadVariableOp?/dense3/kernel/Regularizer/Square/ReadVariableOp?dense4/BiasAdd/ReadVariableOp?dense4/MatMul/ReadVariableOp?
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype02
dense1/MatMul/ReadVariableOp?
dense1/MatMulMatMulinputs$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense1/MatMul?
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense1/BiasAdd/ReadVariableOp?
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense1/BiasAddm
dense1/ReluReludense1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense1/Relu?
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
dense2/MatMul/ReadVariableOp?
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense2/MatMul?
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense2/BiasAdd/ReadVariableOp?
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense2/BiasAddn
dense2/ReluReludense2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense2/Relu?
dense3/MatMul/ReadVariableOpReadVariableOp%dense3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense3/MatMul/ReadVariableOp?
dense3/MatMulMatMuldense2/Relu:activations:0$dense3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense3/MatMul?
dense3/BiasAdd/ReadVariableOpReadVariableOp&dense3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense3/BiasAdd/ReadVariableOp?
dense3/BiasAddBiasAdddense3/MatMul:product:0%dense3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense3/BiasAddn
dense3/ReluReludense3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense3/Relu?
dense4/MatMul/ReadVariableOpReadVariableOp%dense4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense4/MatMul/ReadVariableOp?
dense4/MatMulMatMuldense3/Relu:activations:0$dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense4/MatMul?
dense4/BiasAdd/ReadVariableOpReadVariableOp&dense4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense4/BiasAdd/ReadVariableOp?
dense4/BiasAddBiasAdddense4/MatMul:product:0%dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense4/BiasAddv
dense4/SigmoidSigmoiddense4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense4/Sigmoid?
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp?
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square?
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const?
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum?
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x?
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mul?
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp?
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2"
 dense2/kernel/Regularizer/Square?
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const?
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum?
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x?
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/mul?
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp?
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2"
 dense3/kernel/Regularizer/Square?
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const?
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum?
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x?
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/mulm
IdentityIdentitydense4/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp0^dense1/kernel/Regularizer/Square/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp0^dense2/kernel/Regularizer/Square/ReadVariableOp^dense3/BiasAdd/ReadVariableOp^dense3/MatMul/ReadVariableOp0^dense3/kernel/Regularizer/Square/ReadVariableOp^dense4/BiasAdd/ReadVariableOp^dense4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????	: : : : : : : : 2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2b
/dense1/kernel/Regularizer/Square/ReadVariableOp/dense1/kernel/Regularizer/Square/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp2b
/dense2/kernel/Regularizer/Square/ReadVariableOp/dense2/kernel/Regularizer/Square/ReadVariableOp2>
dense3/BiasAdd/ReadVariableOpdense3/BiasAdd/ReadVariableOp2<
dense3/MatMul/ReadVariableOpdense3/MatMul/ReadVariableOp2b
/dense3/kernel/Regularizer/Square/ReadVariableOp/dense3/kernel/Regularizer/Square/ReadVariableOp2>
dense4/BiasAdd/ReadVariableOpdense4/BiasAdd/ReadVariableOp2<
dense4/MatMul/ReadVariableOpdense4/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?3
?
?__inference_model_layer_call_and_return_conditional_losses_4295

inputs
dense1_4208:	@
dense1_4210:@
dense2_4231:	@?
dense2_4233:	?
dense3_4254:
??
dense3_4256:	?
dense4_4271:	?
dense4_4273:
identity??dense1/StatefulPartitionedCall?/dense1/kernel/Regularizer/Square/ReadVariableOp?dense2/StatefulPartitionedCall?/dense2/kernel/Regularizer/Square/ReadVariableOp?dense3/StatefulPartitionedCall?/dense3/kernel/Regularizer/Square/ReadVariableOp?dense4/StatefulPartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_4208dense1_4210*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_42072 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_4231dense2_4233*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_42302 
dense2/StatefulPartitionedCall?
dense3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0dense3_4254dense3_4256*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense3_layer_call_and_return_conditional_losses_42532 
dense3/StatefulPartitionedCall?
dense4/StatefulPartitionedCallStatefulPartitionedCall'dense3/StatefulPartitionedCall:output:0dense4_4271dense4_4273*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense4_layer_call_and_return_conditional_losses_42702 
dense4/StatefulPartitionedCall?
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense1_4208*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp?
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square?
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const?
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum?
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x?
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mul?
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense2_4231*
_output_shapes
:	@?*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp?
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2"
 dense2/kernel/Regularizer/Square?
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const?
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum?
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x?
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/mul?
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense3_4254* 
_output_shapes
:
??*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp?
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2"
 dense3/kernel/Regularizer/Square?
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const?
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum?
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x?
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/mul?
IdentityIdentity'dense4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense1/StatefulPartitionedCall0^dense1/kernel/Regularizer/Square/ReadVariableOp^dense2/StatefulPartitionedCall0^dense2/kernel/Regularizer/Square/ReadVariableOp^dense3/StatefulPartitionedCall0^dense3/kernel/Regularizer/Square/ReadVariableOp^dense4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????	: : : : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2b
/dense1/kernel/Regularizer/Square/ReadVariableOp/dense1/kernel/Regularizer/Square/ReadVariableOp2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2b
/dense2/kernel/Regularizer/Square/ReadVariableOp/dense2/kernel/Regularizer/Square/ReadVariableOp2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2b
/dense3/kernel/Regularizer/Square/ReadVariableOp/dense3/kernel/Regularizer/Square/ReadVariableOp2@
dense4/StatefulPartitionedCalldense4/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
%__inference_dense1_layer_call_fn_4764

inputs
unknown:	@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_42072
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?D
?
?__inference_model_layer_call_and_return_conditional_losses_4640

inputs7
%dense1_matmul_readvariableop_resource:	@4
&dense1_biasadd_readvariableop_resource:@8
%dense2_matmul_readvariableop_resource:	@?5
&dense2_biasadd_readvariableop_resource:	?9
%dense3_matmul_readvariableop_resource:
??5
&dense3_biasadd_readvariableop_resource:	?8
%dense4_matmul_readvariableop_resource:	?4
&dense4_biasadd_readvariableop_resource:
identity??dense1/BiasAdd/ReadVariableOp?dense1/MatMul/ReadVariableOp?/dense1/kernel/Regularizer/Square/ReadVariableOp?dense2/BiasAdd/ReadVariableOp?dense2/MatMul/ReadVariableOp?/dense2/kernel/Regularizer/Square/ReadVariableOp?dense3/BiasAdd/ReadVariableOp?dense3/MatMul/ReadVariableOp?/dense3/kernel/Regularizer/Square/ReadVariableOp?dense4/BiasAdd/ReadVariableOp?dense4/MatMul/ReadVariableOp?
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype02
dense1/MatMul/ReadVariableOp?
dense1/MatMulMatMulinputs$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense1/MatMul?
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense1/BiasAdd/ReadVariableOp?
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense1/BiasAddm
dense1/ReluReludense1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense1/Relu?
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
dense2/MatMul/ReadVariableOp?
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense2/MatMul?
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense2/BiasAdd/ReadVariableOp?
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense2/BiasAddn
dense2/ReluReludense2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense2/Relu?
dense3/MatMul/ReadVariableOpReadVariableOp%dense3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense3/MatMul/ReadVariableOp?
dense3/MatMulMatMuldense2/Relu:activations:0$dense3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense3/MatMul?
dense3/BiasAdd/ReadVariableOpReadVariableOp&dense3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense3/BiasAdd/ReadVariableOp?
dense3/BiasAddBiasAdddense3/MatMul:product:0%dense3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense3/BiasAddn
dense3/ReluReludense3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense3/Relu?
dense4/MatMul/ReadVariableOpReadVariableOp%dense4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense4/MatMul/ReadVariableOp?
dense4/MatMulMatMuldense3/Relu:activations:0$dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense4/MatMul?
dense4/BiasAdd/ReadVariableOpReadVariableOp&dense4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense4/BiasAdd/ReadVariableOp?
dense4/BiasAddBiasAdddense4/MatMul:product:0%dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense4/BiasAddv
dense4/SigmoidSigmoiddense4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense4/Sigmoid?
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp?
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square?
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const?
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum?
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x?
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mul?
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp?
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2"
 dense2/kernel/Regularizer/Square?
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const?
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum?
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x?
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/mul?
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp?
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2"
 dense3/kernel/Regularizer/Square?
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const?
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum?
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x?
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/mulm
IdentityIdentitydense4/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp0^dense1/kernel/Regularizer/Square/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp0^dense2/kernel/Regularizer/Square/ReadVariableOp^dense3/BiasAdd/ReadVariableOp^dense3/MatMul/ReadVariableOp0^dense3/kernel/Regularizer/Square/ReadVariableOp^dense4/BiasAdd/ReadVariableOp^dense4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????	: : : : : : : : 2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2b
/dense1/kernel/Regularizer/Square/ReadVariableOp/dense1/kernel/Regularizer/Square/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp2b
/dense2/kernel/Regularizer/Square/ReadVariableOp/dense2/kernel/Regularizer/Square/ReadVariableOp2>
dense3/BiasAdd/ReadVariableOpdense3/BiasAdd/ReadVariableOp2<
dense3/MatMul/ReadVariableOpdense3/MatMul/ReadVariableOp2b
/dense3/kernel/Regularizer/Square/ReadVariableOp/dense3/kernel/Regularizer/Square/ReadVariableOp2>
dense4/BiasAdd/ReadVariableOpdense4/BiasAdd/ReadVariableOp2<
dense4/MatMul/ReadVariableOpdense4/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
@__inference_dense1_layer_call_and_return_conditional_losses_4755

inputs0
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?/dense1/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp?
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square?
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const?
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum?
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x?
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense1/kernel/Regularizer/Square/ReadVariableOp/dense1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
%__inference_dense2_layer_call_fn_4796

inputs
unknown:	@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_42302
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_4859J
8dense1_kernel_regularizer_square_readvariableop_resource:	@
identity??/dense1/kernel/Regularizer/Square/ReadVariableOp?
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense1_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp?
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square?
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const?
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum?
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x?
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mulk
IdentityIdentity!dense1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp0^dense1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense1/kernel/Regularizer/Square/ReadVariableOp/dense1/kernel/Regularizer/Square/ReadVariableOp
?
?
@__inference_dense2_layer_call_and_return_conditional_losses_4787

inputs1
matmul_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?/dense2/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp?
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2"
 dense2/kernel/Regularizer/Square?
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const?
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum?
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x?
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/muln
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense2/kernel/Regularizer/Square/ReadVariableOp/dense2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
 __inference__traced_restore_5112
file_prefix0
assignvariableop_dense1_kernel:	@,
assignvariableop_1_dense1_bias:@3
 assignvariableop_2_dense2_kernel:	@?-
assignvariableop_3_dense2_bias:	?4
 assignvariableop_4_dense3_kernel:
??-
assignvariableop_5_dense3_bias:	?3
 assignvariableop_6_dense4_kernel:	?,
assignvariableop_7_dense4_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: :
(assignvariableop_17_adam_dense1_kernel_m:	@4
&assignvariableop_18_adam_dense1_bias_m:@;
(assignvariableop_19_adam_dense2_kernel_m:	@?5
&assignvariableop_20_adam_dense2_bias_m:	?<
(assignvariableop_21_adam_dense3_kernel_m:
??5
&assignvariableop_22_adam_dense3_bias_m:	?;
(assignvariableop_23_adam_dense4_kernel_m:	?4
&assignvariableop_24_adam_dense4_bias_m::
(assignvariableop_25_adam_dense1_kernel_v:	@4
&assignvariableop_26_adam_dense1_bias_v:@;
(assignvariableop_27_adam_dense2_kernel_v:	@?5
&assignvariableop_28_adam_dense2_bias_v:	?<
(assignvariableop_29_adam_dense3_kernel_v:
??5
&assignvariableop_30_adam_dense3_bias_v:	?;
(assignvariableop_31_adam_dense4_kernel_v:	?4
&assignvariableop_32_adam_dense4_bias_v:
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_dense1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_dense2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_dense3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense4_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_dense4_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_dense1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_dense1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense2_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_dense2_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense3_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_dense3_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_dense4_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_dense4_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33f
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_34?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
@__inference_dense2_layer_call_and_return_conditional_losses_4230

inputs1
matmul_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?/dense2/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp?
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2"
 dense2/kernel/Regularizer/Square?
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const?
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum?
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x?
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/muln
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense2/kernel/Regularizer/Square/ReadVariableOp/dense2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?3
?
?__inference_model_layer_call_and_return_conditional_losses_4543

input1
dense1_4504:	@
dense1_4506:@
dense2_4509:	@?
dense2_4511:	?
dense3_4514:
??
dense3_4516:	?
dense4_4519:	?
dense4_4521:
identity??dense1/StatefulPartitionedCall?/dense1/kernel/Regularizer/Square/ReadVariableOp?dense2/StatefulPartitionedCall?/dense2/kernel/Regularizer/Square/ReadVariableOp?dense3/StatefulPartitionedCall?/dense3/kernel/Regularizer/Square/ReadVariableOp?dense4/StatefulPartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCallinput1dense1_4504dense1_4506*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_42072 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_4509dense2_4511*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_42302 
dense2/StatefulPartitionedCall?
dense3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0dense3_4514dense3_4516*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense3_layer_call_and_return_conditional_losses_42532 
dense3/StatefulPartitionedCall?
dense4/StatefulPartitionedCallStatefulPartitionedCall'dense3/StatefulPartitionedCall:output:0dense4_4519dense4_4521*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense4_layer_call_and_return_conditional_losses_42702 
dense4/StatefulPartitionedCall?
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense1_4504*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp?
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square?
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const?
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum?
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x?
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mul?
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense2_4509*
_output_shapes
:	@?*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp?
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2"
 dense2/kernel/Regularizer/Square?
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const?
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum?
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x?
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/mul?
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense3_4514* 
_output_shapes
:
??*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp?
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2"
 dense3/kernel/Regularizer/Square?
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const?
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum?
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x?
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/mul?
IdentityIdentity'dense4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense1/StatefulPartitionedCall0^dense1/kernel/Regularizer/Square/ReadVariableOp^dense2/StatefulPartitionedCall0^dense2/kernel/Regularizer/Square/ReadVariableOp^dense3/StatefulPartitionedCall0^dense3/kernel/Regularizer/Square/ReadVariableOp^dense4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????	: : : : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2b
/dense1/kernel/Regularizer/Square/ReadVariableOp/dense1/kernel/Regularizer/Square/ReadVariableOp2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2b
/dense2/kernel/Regularizer/Square/ReadVariableOp/dense2/kernel/Regularizer/Square/ReadVariableOp2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2b
/dense3/kernel/Regularizer/Square/ReadVariableOp/dense3/kernel/Regularizer/Square/ReadVariableOp2@
dense4/StatefulPartitionedCalldense4/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinput1
?	
?
$__inference_model_layer_call_fn_4732

inputs
unknown:	@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_44192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
@__inference_dense3_layer_call_and_return_conditional_losses_4819

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?/dense3/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp?
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2"
 dense3/kernel/Regularizer/Square?
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const?
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum?
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x?
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/muln
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense3/kernel/Regularizer/Square/ReadVariableOp/dense3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?3
?
?__inference_model_layer_call_and_return_conditional_losses_4501

input1
dense1_4462:	@
dense1_4464:@
dense2_4467:	@?
dense2_4469:	?
dense3_4472:
??
dense3_4474:	?
dense4_4477:	?
dense4_4479:
identity??dense1/StatefulPartitionedCall?/dense1/kernel/Regularizer/Square/ReadVariableOp?dense2/StatefulPartitionedCall?/dense2/kernel/Regularizer/Square/ReadVariableOp?dense3/StatefulPartitionedCall?/dense3/kernel/Regularizer/Square/ReadVariableOp?dense4/StatefulPartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCallinput1dense1_4462dense1_4464*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_42072 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_4467dense2_4469*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_42302 
dense2/StatefulPartitionedCall?
dense3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0dense3_4472dense3_4474*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense3_layer_call_and_return_conditional_losses_42532 
dense3/StatefulPartitionedCall?
dense4/StatefulPartitionedCallStatefulPartitionedCall'dense3/StatefulPartitionedCall:output:0dense4_4477dense4_4479*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense4_layer_call_and_return_conditional_losses_42702 
dense4/StatefulPartitionedCall?
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense1_4462*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp?
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square?
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const?
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum?
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x?
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mul?
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense2_4467*
_output_shapes
:	@?*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp?
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?2"
 dense2/kernel/Regularizer/Square?
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const?
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum?
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x?
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/mul?
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense3_4472* 
_output_shapes
:
??*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp?
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2"
 dense3/kernel/Regularizer/Square?
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const?
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum?
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x?
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/mul?
IdentityIdentity'dense4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense1/StatefulPartitionedCall0^dense1/kernel/Regularizer/Square/ReadVariableOp^dense2/StatefulPartitionedCall0^dense2/kernel/Regularizer/Square/ReadVariableOp^dense3/StatefulPartitionedCall0^dense3/kernel/Regularizer/Square/ReadVariableOp^dense4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????	: : : : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2b
/dense1/kernel/Regularizer/Square/ReadVariableOp/dense1/kernel/Regularizer/Square/ReadVariableOp2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2b
/dense2/kernel/Regularizer/Square/ReadVariableOp/dense2/kernel/Regularizer/Square/ReadVariableOp2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2b
/dense3/kernel/Regularizer/Square/ReadVariableOp/dense3/kernel/Regularizer/Square/ReadVariableOp2@
dense4/StatefulPartitionedCalldense4/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinput1
?	
?
$__inference_model_layer_call_fn_4459

input1
unknown:	@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_44192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinput1
?
?
%__inference_dense4_layer_call_fn_4848

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense4_layer_call_and_return_conditional_losses_42702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
$__inference_model_layer_call_fn_4711

inputs
unknown:	@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_42952
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
9
input1/
serving_default_input1:0?????????	:
dense40
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?b
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
*]&call_and_return_all_conditional_losses
^__call__
__default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*`&call_and_return_all_conditional_losses
a__call__"
_tf_keras_layer
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*b&call_and_return_all_conditional_losses
c__call__"
_tf_keras_layer
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*d&call_and_return_all_conditional_losses
e__call__"
_tf_keras_layer
?

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
*f&call_and_return_all_conditional_losses
g__call__"
_tf_keras_layer
?
$iter

%beta_1

&beta_2
	'decay
(learning_ratemMmNmOmPmQmRmSmTvUvVvWvXvYvZv[v\"
	optimizer
5
h0
i1
j2"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
?
regularization_losses
	variables
)non_trainable_variables
	trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
^__call__
__default_save_signature
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
,
kserving_default"
signature_map
:	@2dense1/kernel
:@2dense1/bias
'
h0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
	variables
.non_trainable_variables
trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 :	@?2dense2/kernel
:?2dense2/bias
'
i0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
	variables
3non_trainable_variables
trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
!:
??2dense3/kernel
:?2dense3/bias
'
j0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
	variables
8non_trainable_variables
trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
 :	?2dense4/kernel
:2dense4/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
 regularization_losses
!	variables
=non_trainable_variables
"trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
h0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
i0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
j0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Dtotal
	Ecount
F	variables
G	keras_api"
_tf_keras_metric
^
	Htotal
	Icount
J
_fn_kwargs
K	variables
L	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
D0
E1"
trackable_list_wrapper
-
F	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
H0
I1"
trackable_list_wrapper
-
K	variables"
_generic_user_object
$:"	@2Adam/dense1/kernel/m
:@2Adam/dense1/bias/m
%:#	@?2Adam/dense2/kernel/m
:?2Adam/dense2/bias/m
&:$
??2Adam/dense3/kernel/m
:?2Adam/dense3/bias/m
%:#	?2Adam/dense4/kernel/m
:2Adam/dense4/bias/m
$:"	@2Adam/dense1/kernel/v
:@2Adam/dense1/bias/v
%:#	@?2Adam/dense2/kernel/v
:?2Adam/dense2/bias/v
&:$
??2Adam/dense3/kernel/v
:?2Adam/dense3/bias/v
%:#	?2Adam/dense4/kernel/v
:2Adam/dense4/bias/v
?2?
?__inference_model_layer_call_and_return_conditional_losses_4640
?__inference_model_layer_call_and_return_conditional_losses_4690
?__inference_model_layer_call_and_return_conditional_losses_4501
?__inference_model_layer_call_and_return_conditional_losses_4543?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference_model_layer_call_fn_4314
$__inference_model_layer_call_fn_4711
$__inference_model_layer_call_fn_4732
$__inference_model_layer_call_fn_4459?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference__wrapped_model_4183input1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense1_layer_call_and_return_conditional_losses_4755?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense1_layer_call_fn_4764?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense2_layer_call_and_return_conditional_losses_4787?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense2_layer_call_fn_4796?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense3_layer_call_and_return_conditional_losses_4819?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense3_layer_call_fn_4828?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense4_layer_call_and_return_conditional_losses_4839?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense4_layer_call_fn_4848?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_4859?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_4870?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_4881?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
"__inference_signature_wrapper_4590input1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_4183l/?,
%?"
 ?
input1?????????	
? "/?,
*
dense4 ?
dense4??????????
@__inference_dense1_layer_call_and_return_conditional_losses_4755\/?,
%?"
 ?
inputs?????????	
? "%?"
?
0?????????@
? x
%__inference_dense1_layer_call_fn_4764O/?,
%?"
 ?
inputs?????????	
? "??????????@?
@__inference_dense2_layer_call_and_return_conditional_losses_4787]/?,
%?"
 ?
inputs?????????@
? "&?#
?
0??????????
? y
%__inference_dense2_layer_call_fn_4796P/?,
%?"
 ?
inputs?????????@
? "????????????
@__inference_dense3_layer_call_and_return_conditional_losses_4819^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? z
%__inference_dense3_layer_call_fn_4828Q0?-
&?#
!?
inputs??????????
? "????????????
@__inference_dense4_layer_call_and_return_conditional_losses_4839]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? y
%__inference_dense4_layer_call_fn_4848P0?-
&?#
!?
inputs??????????
? "??????????9
__inference_loss_fn_0_4859?

? 
? "? 9
__inference_loss_fn_1_4870?

? 
? "? 9
__inference_loss_fn_2_4881?

? 
? "? ?
?__inference_model_layer_call_and_return_conditional_losses_4501j7?4
-?*
 ?
input1?????????	
p 

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_4543j7?4
-?*
 ?
input1?????????	
p

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_4640j7?4
-?*
 ?
inputs?????????	
p 

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_4690j7?4
-?*
 ?
inputs?????????	
p

 
? "%?"
?
0?????????
? ?
$__inference_model_layer_call_fn_4314]7?4
-?*
 ?
input1?????????	
p 

 
? "???????????
$__inference_model_layer_call_fn_4459]7?4
-?*
 ?
input1?????????	
p

 
? "???????????
$__inference_model_layer_call_fn_4711]7?4
-?*
 ?
inputs?????????	
p 

 
? "???????????
$__inference_model_layer_call_fn_4732]7?4
-?*
 ?
inputs?????????	
p

 
? "???????????
"__inference_signature_wrapper_4590v9?6
? 
/?,
*
input1 ?
input1?????????	"/?,
*
dense4 ?
dense4?????????