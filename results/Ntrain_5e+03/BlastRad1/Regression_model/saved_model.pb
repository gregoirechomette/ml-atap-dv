ñä
Ô
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
@
Softplus
features"T
activations"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
¾
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02v2.6.0-rc2-32-g919f693420e8ÿ
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
v
dense5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	@*
shared_namedense5/kernel
o
!dense5/kernel/Read/ReadVariableOpReadVariableOpdense5/kernel*
_output_shapes

:	@*
dtype0
n
dense5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense5/bias
g
dense5/bias/Read/ReadVariableOpReadVariableOpdense5/bias*
_output_shapes
:@*
dtype0
w
dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense2/kernel
p
!dense2/kernel/Read/ReadVariableOpReadVariableOpdense2/kernel*
_output_shapes
:	@*
dtype0
o
dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense2/bias
h
dense2/bias/Read/ReadVariableOpReadVariableOpdense2/bias*
_output_shapes	
:*
dtype0
w
dense6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense6/kernel
p
!dense6/kernel/Read/ReadVariableOpReadVariableOpdense6/kernel*
_output_shapes
:	@*
dtype0
o
dense6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense6/bias
h
dense6/bias/Read/ReadVariableOpReadVariableOpdense6/bias*
_output_shapes	
:*
dtype0
x
dense3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense3/kernel
q
!dense3/kernel/Read/ReadVariableOpReadVariableOpdense3/kernel* 
_output_shapes
:
*
dtype0
o
dense3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense3/bias
h
dense3/bias/Read/ReadVariableOpReadVariableOpdense3/bias*
_output_shapes	
:*
dtype0
x
dense7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense7/kernel
q
!dense7/kernel/Read/ReadVariableOpReadVariableOpdense7/kernel* 
_output_shapes
:
*
dtype0
o
dense7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense7/bias
h
dense7/bias/Read/ReadVariableOpReadVariableOpdense7/bias*
_output_shapes	
:*
dtype0
w
dense4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense4/kernel
p
!dense4/kernel/Read/ReadVariableOpReadVariableOpdense4/kernel*
_output_shapes
:	*
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
w
dense8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense8/kernel
p
!dense8/kernel/Read/ReadVariableOpReadVariableOpdense8/kernel*
_output_shapes
:	*
dtype0
n
dense8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense8/bias
g
dense8/bias/Read/ReadVariableOpReadVariableOpdense8/bias*
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

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

Adam/dense5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	@*%
shared_nameAdam/dense5/kernel/m
}
(Adam/dense5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense5/kernel/m*
_output_shapes

:	@*
dtype0
|
Adam/dense5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/dense5/bias/m
u
&Adam/dense5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense5/bias/m*
_output_shapes
:@*
dtype0

Adam/dense2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*%
shared_nameAdam/dense2/kernel/m
~
(Adam/dense2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense2/kernel/m*
_output_shapes
:	@*
dtype0
}
Adam/dense2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/dense2/bias/m
v
&Adam/dense2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense2/bias/m*
_output_shapes	
:*
dtype0

Adam/dense6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*%
shared_nameAdam/dense6/kernel/m
~
(Adam/dense6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense6/kernel/m*
_output_shapes
:	@*
dtype0
}
Adam/dense6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/dense6/bias/m
v
&Adam/dense6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense6/bias/m*
_output_shapes	
:*
dtype0

Adam/dense3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense3/kernel/m

(Adam/dense3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense3/kernel/m* 
_output_shapes
:
*
dtype0
}
Adam/dense3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/dense3/bias/m
v
&Adam/dense3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense3/bias/m*
_output_shapes	
:*
dtype0

Adam/dense7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense7/kernel/m

(Adam/dense7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense7/kernel/m* 
_output_shapes
:
*
dtype0
}
Adam/dense7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/dense7/bias/m
v
&Adam/dense7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense7/bias/m*
_output_shapes	
:*
dtype0

Adam/dense4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/dense4/kernel/m
~
(Adam/dense4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense4/kernel/m*
_output_shapes
:	*
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

Adam/dense8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/dense8/kernel/m
~
(Adam/dense8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense8/kernel/m*
_output_shapes
:	*
dtype0
|
Adam/dense8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/dense8/bias/m
u
&Adam/dense8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense8/bias/m*
_output_shapes
:*
dtype0

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

Adam/dense5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	@*%
shared_nameAdam/dense5/kernel/v
}
(Adam/dense5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense5/kernel/v*
_output_shapes

:	@*
dtype0
|
Adam/dense5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/dense5/bias/v
u
&Adam/dense5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense5/bias/v*
_output_shapes
:@*
dtype0

Adam/dense2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*%
shared_nameAdam/dense2/kernel/v
~
(Adam/dense2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense2/kernel/v*
_output_shapes
:	@*
dtype0
}
Adam/dense2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/dense2/bias/v
v
&Adam/dense2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense2/bias/v*
_output_shapes	
:*
dtype0

Adam/dense6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*%
shared_nameAdam/dense6/kernel/v
~
(Adam/dense6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense6/kernel/v*
_output_shapes
:	@*
dtype0
}
Adam/dense6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/dense6/bias/v
v
&Adam/dense6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense6/bias/v*
_output_shapes	
:*
dtype0

Adam/dense3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense3/kernel/v

(Adam/dense3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense3/kernel/v* 
_output_shapes
:
*
dtype0
}
Adam/dense3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/dense3/bias/v
v
&Adam/dense3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense3/bias/v*
_output_shapes	
:*
dtype0

Adam/dense7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense7/kernel/v

(Adam/dense7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense7/kernel/v* 
_output_shapes
:
*
dtype0
}
Adam/dense7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/dense7/bias/v
v
&Adam/dense7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense7/bias/v*
_output_shapes	
:*
dtype0

Adam/dense4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/dense4/kernel/v
~
(Adam/dense4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense4/kernel/v*
_output_shapes
:	*
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

Adam/dense8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/dense8/kernel/v
~
(Adam/dense8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense8/kernel/v*
_output_shapes
:	*
dtype0
|
Adam/dense8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/dense8/bias/v
u
&Adam/dense8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense8/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
S
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÏR
valueÅRBÂR B»R
¯
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
	optimizer
loss
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
h

%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
h

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
h

7kernel
8bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
 
 
h

=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
h

Ckernel
Dbias
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api

I	keras_api

J	keras_api

K	keras_api

L	keras_api

M	keras_api
R
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api

Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratemmm m%m&m+m,m1m2m7m8m=m>mCmDmvvv  v¡%v¢&v£+v¤,v¥1v¦2v§7v¨8v©=vª>v«Cv¬Dv­
 
 
v
0
1
2
 3
%4
&5
+6
,7
18
29
710
811
=12
>13
C14
D15
v
0
1
2
 3
%4
&5
+6
,7
18
29
710
811
=12
>13
C14
D15
­
regularization_losses
	variables
Wnon_trainable_variables
trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
 
YW
VARIABLE_VALUEdense1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
	variables
\non_trainable_variables
trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
YW
VARIABLE_VALUEdense5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
­
!regularization_losses
"	variables
anon_trainable_variables
#trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
YW
VARIABLE_VALUEdense2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
­
'regularization_losses
(	variables
fnon_trainable_variables
)trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
YW
VARIABLE_VALUEdense6/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense6/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
­
-regularization_losses
.	variables
knon_trainable_variables
/trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
YW
VARIABLE_VALUEdense3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
­
3regularization_losses
4	variables
pnon_trainable_variables
5trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
YW
VARIABLE_VALUEdense7/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense7/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

70
81

70
81
­
9regularization_losses
:	variables
unon_trainable_variables
;trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
YW
VARIABLE_VALUEdense4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
­
?regularization_losses
@	variables
znon_trainable_variables
Atrainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
YW
VARIABLE_VALUEdense8/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense8/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1

C0
D1
±
Eregularization_losses
F	variables
non_trainable_variables
Gtrainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 
 
 
 
 
 
 
 
²
Nregularization_losses
O	variables
non_trainable_variables
Ptrainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
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
~
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16

0
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
 
 
 
8

total

count
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
|z
VARIABLE_VALUEAdam/dense1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense5/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense5/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense6/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense6/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense7/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense7/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense4/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense4/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense8/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense8/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense5/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense5/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense6/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense6/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense7/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense7/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense4/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense4/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense8/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense8/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_input1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ	
y
serving_default_input2Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
y
serving_default_input3Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
õ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input1serving_default_input2serving_default_input3dense5/kerneldense5/biasdense1/kerneldense1/biasdense6/kerneldense6/biasdense2/kerneldense2/biasdense7/kerneldense7/biasdense3/kerneldense3/biasdense8/kerneldense8/biasdense4/kerneldense4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_8641
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ï
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp!dense5/kernel/Read/ReadVariableOpdense5/bias/Read/ReadVariableOp!dense2/kernel/Read/ReadVariableOpdense2/bias/Read/ReadVariableOp!dense6/kernel/Read/ReadVariableOpdense6/bias/Read/ReadVariableOp!dense3/kernel/Read/ReadVariableOpdense3/bias/Read/ReadVariableOp!dense7/kernel/Read/ReadVariableOpdense7/bias/Read/ReadVariableOp!dense4/kernel/Read/ReadVariableOpdense4/bias/Read/ReadVariableOp!dense8/kernel/Read/ReadVariableOpdense8/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/dense1/kernel/m/Read/ReadVariableOp&Adam/dense1/bias/m/Read/ReadVariableOp(Adam/dense5/kernel/m/Read/ReadVariableOp&Adam/dense5/bias/m/Read/ReadVariableOp(Adam/dense2/kernel/m/Read/ReadVariableOp&Adam/dense2/bias/m/Read/ReadVariableOp(Adam/dense6/kernel/m/Read/ReadVariableOp&Adam/dense6/bias/m/Read/ReadVariableOp(Adam/dense3/kernel/m/Read/ReadVariableOp&Adam/dense3/bias/m/Read/ReadVariableOp(Adam/dense7/kernel/m/Read/ReadVariableOp&Adam/dense7/bias/m/Read/ReadVariableOp(Adam/dense4/kernel/m/Read/ReadVariableOp&Adam/dense4/bias/m/Read/ReadVariableOp(Adam/dense8/kernel/m/Read/ReadVariableOp&Adam/dense8/bias/m/Read/ReadVariableOp(Adam/dense1/kernel/v/Read/ReadVariableOp&Adam/dense1/bias/v/Read/ReadVariableOp(Adam/dense5/kernel/v/Read/ReadVariableOp&Adam/dense5/bias/v/Read/ReadVariableOp(Adam/dense2/kernel/v/Read/ReadVariableOp&Adam/dense2/bias/v/Read/ReadVariableOp(Adam/dense6/kernel/v/Read/ReadVariableOp&Adam/dense6/bias/v/Read/ReadVariableOp(Adam/dense3/kernel/v/Read/ReadVariableOp&Adam/dense3/bias/v/Read/ReadVariableOp(Adam/dense7/kernel/v/Read/ReadVariableOp&Adam/dense7/bias/v/Read/ReadVariableOp(Adam/dense4/kernel/v/Read/ReadVariableOp&Adam/dense4/bias/v/Read/ReadVariableOp(Adam/dense8/kernel/v/Read/ReadVariableOp&Adam/dense8/bias/v/Read/ReadVariableOpConst*D
Tin=
;29	*
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
GPU 2J 8 *&
f!R
__inference__traced_save_9434


StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense1/kerneldense1/biasdense5/kerneldense5/biasdense2/kerneldense2/biasdense6/kerneldense6/biasdense3/kerneldense3/biasdense7/kerneldense7/biasdense4/kerneldense4/biasdense8/kerneldense8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense1/kernel/mAdam/dense1/bias/mAdam/dense5/kernel/mAdam/dense5/bias/mAdam/dense2/kernel/mAdam/dense2/bias/mAdam/dense6/kernel/mAdam/dense6/bias/mAdam/dense3/kernel/mAdam/dense3/bias/mAdam/dense7/kernel/mAdam/dense7/bias/mAdam/dense4/kernel/mAdam/dense4/bias/mAdam/dense8/kernel/mAdam/dense8/bias/mAdam/dense1/kernel/vAdam/dense1/bias/vAdam/dense5/kernel/vAdam/dense5/bias/vAdam/dense2/kernel/vAdam/dense2/bias/vAdam/dense6/kernel/vAdam/dense6/bias/vAdam/dense3/kernel/vAdam/dense3/bias/vAdam/dense7/kernel/vAdam/dense7/bias/vAdam/dense4/kernel/vAdam/dense4/bias/vAdam/dense8/kernel/vAdam/dense8/bias/v*C
Tin<
:28*
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
GPU 2J 8 *)
f$R"
 __inference__traced_restore_9609ã³
Ü
¥
@__inference_dense2_layer_call_and_return_conditional_losses_7891

inputs1
matmul_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense2/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ReluÂ
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp±
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense2/kernel/Regularizer/Square
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const¶
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x¸
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/muln
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity±
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense2/kernel/Regularizer/Square/ReadVariableOp/dense2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Èr
¨	
A__inference_model_1_layer_call_and_return_conditional_losses_8464

input1

input2

input3
dense5_8377:	@
dense5_8379:@
dense1_8382:	@
dense1_8384:@
dense6_8387:	@
dense6_8389:	
dense2_8392:	@
dense2_8394:	
dense7_8397:

dense7_8399:	
dense3_8402:

dense3_8404:	
dense8_8407:	
dense8_8409:
dense4_8412:	
dense4_8414:
identity

identity_1

identity_2¢dense1/StatefulPartitionedCall¢/dense1/kernel/Regularizer/Square/ReadVariableOp¢dense2/StatefulPartitionedCall¢/dense2/kernel/Regularizer/Square/ReadVariableOp¢dense3/StatefulPartitionedCall¢/dense3/kernel/Regularizer/Square/ReadVariableOp¢dense4/StatefulPartitionedCall¢dense5/StatefulPartitionedCall¢/dense5/kernel/Regularizer/Square/ReadVariableOp¢dense6/StatefulPartitionedCall¢/dense6/kernel/Regularizer/Square/ReadVariableOp¢dense7/StatefulPartitionedCall¢/dense7/kernel/Regularizer/Square/ReadVariableOp¢dense8/StatefulPartitionedCall
dense5/StatefulPartitionedCallStatefulPartitionedCallinput1dense5_8377dense5_8379*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense5_layer_call_and_return_conditional_losses_78222 
dense5/StatefulPartitionedCall
dense1/StatefulPartitionedCallStatefulPartitionedCallinput1dense1_8382dense1_8384*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_78452 
dense1/StatefulPartitionedCall¦
dense6/StatefulPartitionedCallStatefulPartitionedCall'dense5/StatefulPartitionedCall:output:0dense6_8387dense6_8389*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense6_layer_call_and_return_conditional_losses_78682 
dense6/StatefulPartitionedCall¦
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_8392dense2_8394*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_78912 
dense2/StatefulPartitionedCall¦
dense7/StatefulPartitionedCallStatefulPartitionedCall'dense6/StatefulPartitionedCall:output:0dense7_8397dense7_8399*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense7_layer_call_and_return_conditional_losses_79142 
dense7/StatefulPartitionedCall¦
dense3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0dense3_8402dense3_8404*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense3_layer_call_and_return_conditional_losses_79372 
dense3/StatefulPartitionedCall¥
dense8/StatefulPartitionedCallStatefulPartitionedCall'dense7/StatefulPartitionedCall:output:0dense8_8407dense8_8409*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense8_layer_call_and_return_conditional_losses_79542 
dense8/StatefulPartitionedCall¥
dense4/StatefulPartitionedCallStatefulPartitionedCall'dense3/StatefulPartitionedCall:output:0dense4_8412dense4_8414*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense4_layer_call_and_return_conditional_losses_79702 
dense4/StatefulPartitionedCall
tf.math.log/LogLog'dense8/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.log/LogÔ
,tf.math.squared_difference/SquaredDifferenceSquaredDifference'dense4/StatefulPartitionedCall:output:0input2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,tf.math.squared_difference/SquaredDifferenceÊ
tf.math.truediv/truedivRealDiv0tf.math.squared_difference/SquaredDifference:z:0'dense8/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.truediv/truediv¥
tf.__operators__.add/AddV2AddV2tf.math.truediv/truediv:z:0tf.math.log/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add/AddV2
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean/Const¡
tf.math.reduce_mean/MeanMeantf.__operators__.add/AddV2:z:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean/Meanß
add_loss/PartitionedCallPartitionedCall!tf.math.reduce_mean/Mean:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_add_loss_layer_call_and_return_conditional_losses_79872
add_loss/PartitionedCall®
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense1_8382*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp°
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const¶
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x¸
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mul®
/dense5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense5_8377*
_output_shapes

:	@*
dtype021
/dense5/kernel/Regularizer/Square/ReadVariableOp°
 dense5/kernel/Regularizer/SquareSquare7dense5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense5/kernel/Regularizer/Square
dense5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense5/kernel/Regularizer/Const¶
dense5/kernel/Regularizer/SumSum$dense5/kernel/Regularizer/Square:y:0(dense5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/Sum
dense5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense5/kernel/Regularizer/mul/x¸
dense5/kernel/Regularizer/mulMul(dense5/kernel/Regularizer/mul/x:output:0&dense5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/mul¯
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense2_8392*
_output_shapes
:	@*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp±
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense2/kernel/Regularizer/Square
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const¶
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x¸
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/mul¯
/dense6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense6_8387*
_output_shapes
:	@*
dtype021
/dense6/kernel/Regularizer/Square/ReadVariableOp±
 dense6/kernel/Regularizer/SquareSquare7dense6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense6/kernel/Regularizer/Square
dense6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense6/kernel/Regularizer/Const¶
dense6/kernel/Regularizer/SumSum$dense6/kernel/Regularizer/Square:y:0(dense6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/Sum
dense6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense6/kernel/Regularizer/mul/x¸
dense6/kernel/Regularizer/mulMul(dense6/kernel/Regularizer/mul/x:output:0&dense6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/mul°
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense3_8402* 
_output_shapes
:
*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp²
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense3/kernel/Regularizer/Square
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const¶
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x¸
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/mul°
/dense7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense7_8397* 
_output_shapes
:
*
dtype021
/dense7/kernel/Regularizer/Square/ReadVariableOp²
 dense7/kernel/Regularizer/SquareSquare7dense7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense7/kernel/Regularizer/Square
dense7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense7/kernel/Regularizer/Const¶
dense7/kernel/Regularizer/SumSum$dense7/kernel/Regularizer/Square:y:0(dense7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/Sum
dense7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense7/kernel/Regularizer/mul/x¸
dense7/kernel/Regularizer/mulMul(dense7/kernel/Regularizer/mul/x:output:0&dense7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/mul
IdentityIdentity'dense4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity'dense8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1o

Identity_2Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 2

Identity_2
NoOpNoOp^dense1/StatefulPartitionedCall0^dense1/kernel/Regularizer/Square/ReadVariableOp^dense2/StatefulPartitionedCall0^dense2/kernel/Regularizer/Square/ReadVariableOp^dense3/StatefulPartitionedCall0^dense3/kernel/Regularizer/Square/ReadVariableOp^dense4/StatefulPartitionedCall^dense5/StatefulPartitionedCall0^dense5/kernel/Regularizer/Square/ReadVariableOp^dense6/StatefulPartitionedCall0^dense6/kernel/Regularizer/Square/ReadVariableOp^dense7/StatefulPartitionedCall0^dense7/kernel/Regularizer/Square/ReadVariableOp^dense8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2b
/dense1/kernel/Regularizer/Square/ReadVariableOp/dense1/kernel/Regularizer/Square/ReadVariableOp2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2b
/dense2/kernel/Regularizer/Square/ReadVariableOp/dense2/kernel/Regularizer/Square/ReadVariableOp2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2b
/dense3/kernel/Regularizer/Square/ReadVariableOp/dense3/kernel/Regularizer/Square/ReadVariableOp2@
dense4/StatefulPartitionedCalldense4/StatefulPartitionedCall2@
dense5/StatefulPartitionedCalldense5/StatefulPartitionedCall2b
/dense5/kernel/Regularizer/Square/ReadVariableOp/dense5/kernel/Regularizer/Square/ReadVariableOp2@
dense6/StatefulPartitionedCalldense6/StatefulPartitionedCall2b
/dense6/kernel/Regularizer/Square/ReadVariableOp/dense6/kernel/Regularizer/Square/ReadVariableOp2@
dense7/StatefulPartitionedCalldense7/StatefulPartitionedCall2b
/dense7/kernel/Regularizer/Square/ReadVariableOp/dense7/kernel/Regularizer/Square/ReadVariableOp2@
dense8/StatefulPartitionedCalldense8/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinput1:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinput2:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinput3
í

%__inference_dense2_layer_call_fn_9031

inputs
unknown:	@
	unknown_0:	
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_78912
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¬
__inference_loss_fn_3_9221K
8dense6_kernel_regularizer_square_readvariableop_resource:	@
identity¢/dense6/kernel/Regularizer/Square/ReadVariableOpÜ
/dense6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense6_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	@*
dtype021
/dense6/kernel/Regularizer/Square/ReadVariableOp±
 dense6/kernel/Regularizer/SquareSquare7dense6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense6/kernel/Regularizer/Square
dense6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense6/kernel/Regularizer/Const¶
dense6/kernel/Regularizer/SumSum$dense6/kernel/Regularizer/Square:y:0(dense6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/Sum
dense6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense6/kernel/Regularizer/mul/x¸
dense6/kernel/Regularizer/mulMul(dense6/kernel/Regularizer/mul/x:output:0&dense6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/mulk
IdentityIdentity!dense6/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
NoOpNoOp0^dense6/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense6/kernel/Regularizer/Square/ReadVariableOp/dense6/kernel/Regularizer/Square/ReadVariableOp
é

%__inference_dense1_layer_call_fn_8967

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_78452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
Ò
£
@__inference_dense5_layer_call_and_return_conditional_losses_8990

inputs0
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense5/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÁ
/dense5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	@*
dtype021
/dense5/kernel/Regularizer/Square/ReadVariableOp°
 dense5/kernel/Regularizer/SquareSquare7dense5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense5/kernel/Regularizer/Square
dense5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense5/kernel/Regularizer/Const¶
dense5/kernel/Regularizer/SumSum$dense5/kernel/Regularizer/Square:y:0(dense5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/Sum
dense5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense5/kernel/Regularizer/mul/x¸
dense5/kernel/Regularizer/mulMul(dense5/kernel/Regularizer/mul/x:output:0&dense5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity±
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense5/kernel/Regularizer/Square/ReadVariableOp/dense5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
â
¦
@__inference_dense3_layer_call_and_return_conditional_losses_7937

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense3/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ReluÃ
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp²
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense3/kernel/Regularizer/Square
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const¶
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x¸
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/muln
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity±
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense3/kernel/Regularizer/Square/ReadVariableOp/dense3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ì
&__inference_model_1_layer_call_fn_8935
inputs_0
inputs_1
inputs_2
unknown:	@
	unknown_0:@
	unknown_1:	@
	unknown_2:@
	unknown_3:	@
	unknown_4:	
	unknown_5:	@
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:

unknown_13:	

unknown_14:
identity

identity_1¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_82922
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2
Ü
¥
@__inference_dense6_layer_call_and_return_conditional_losses_7868

inputs1
matmul_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense6/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ReluÂ
/dense6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype021
/dense6/kernel/Regularizer/Square/ReadVariableOp±
 dense6/kernel/Regularizer/SquareSquare7dense6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense6/kernel/Regularizer/Square
dense6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense6/kernel/Regularizer/Const¶
dense6/kernel/Regularizer/SumSum$dense6/kernel/Regularizer/Square:y:0(dense6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/Sum
dense6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense6/kernel/Regularizer/mul/x¸
dense6/kernel/Regularizer/mulMul(dense6/kernel/Regularizer/mul/x:output:0&dense6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/muln
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity±
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense6/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense6/kernel/Regularizer/Square/ReadVariableOp/dense6/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Í
n
B__inference_add_loss_layer_call_and_return_conditional_losses_9171

inputs
identity

identity_1I
IdentityIdentityinputs*
T0*
_output_shapes
: 2

IdentityM

Identity_1Identityinputs*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs

¬
__inference_loss_fn_2_9210K
8dense2_kernel_regularizer_square_readvariableop_resource:	@
identity¢/dense2/kernel/Regularizer/Square/ReadVariableOpÜ
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense2_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	@*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp±
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense2/kernel/Regularizer/Square
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const¶
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x¸
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/mulk
IdentityIdentity!dense2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
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
á
Â
"__inference_signature_wrapper_8641

input1

input2

input3
unknown:	@
	unknown_0:@
	unknown_1:	@
	unknown_2:@
	unknown_3:	@
	unknown_4:	
	unknown_5:	@
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:

unknown_13:	

unknown_14:
identity

identity_1¢StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinput1input2input3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_77942
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinput1:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinput2:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinput3
Èr
¨	
A__inference_model_1_layer_call_and_return_conditional_losses_8556

input1

input2

input3
dense5_8469:	@
dense5_8471:@
dense1_8474:	@
dense1_8476:@
dense6_8479:	@
dense6_8481:	
dense2_8484:	@
dense2_8486:	
dense7_8489:

dense7_8491:	
dense3_8494:

dense3_8496:	
dense8_8499:	
dense8_8501:
dense4_8504:	
dense4_8506:
identity

identity_1

identity_2¢dense1/StatefulPartitionedCall¢/dense1/kernel/Regularizer/Square/ReadVariableOp¢dense2/StatefulPartitionedCall¢/dense2/kernel/Regularizer/Square/ReadVariableOp¢dense3/StatefulPartitionedCall¢/dense3/kernel/Regularizer/Square/ReadVariableOp¢dense4/StatefulPartitionedCall¢dense5/StatefulPartitionedCall¢/dense5/kernel/Regularizer/Square/ReadVariableOp¢dense6/StatefulPartitionedCall¢/dense6/kernel/Regularizer/Square/ReadVariableOp¢dense7/StatefulPartitionedCall¢/dense7/kernel/Regularizer/Square/ReadVariableOp¢dense8/StatefulPartitionedCall
dense5/StatefulPartitionedCallStatefulPartitionedCallinput1dense5_8469dense5_8471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense5_layer_call_and_return_conditional_losses_78222 
dense5/StatefulPartitionedCall
dense1/StatefulPartitionedCallStatefulPartitionedCallinput1dense1_8474dense1_8476*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_78452 
dense1/StatefulPartitionedCall¦
dense6/StatefulPartitionedCallStatefulPartitionedCall'dense5/StatefulPartitionedCall:output:0dense6_8479dense6_8481*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense6_layer_call_and_return_conditional_losses_78682 
dense6/StatefulPartitionedCall¦
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_8484dense2_8486*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_78912 
dense2/StatefulPartitionedCall¦
dense7/StatefulPartitionedCallStatefulPartitionedCall'dense6/StatefulPartitionedCall:output:0dense7_8489dense7_8491*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense7_layer_call_and_return_conditional_losses_79142 
dense7/StatefulPartitionedCall¦
dense3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0dense3_8494dense3_8496*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense3_layer_call_and_return_conditional_losses_79372 
dense3/StatefulPartitionedCall¥
dense8/StatefulPartitionedCallStatefulPartitionedCall'dense7/StatefulPartitionedCall:output:0dense8_8499dense8_8501*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense8_layer_call_and_return_conditional_losses_79542 
dense8/StatefulPartitionedCall¥
dense4/StatefulPartitionedCallStatefulPartitionedCall'dense3/StatefulPartitionedCall:output:0dense4_8504dense4_8506*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense4_layer_call_and_return_conditional_losses_79702 
dense4/StatefulPartitionedCall
tf.math.log/LogLog'dense8/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.log/LogÔ
,tf.math.squared_difference/SquaredDifferenceSquaredDifference'dense4/StatefulPartitionedCall:output:0input2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,tf.math.squared_difference/SquaredDifferenceÊ
tf.math.truediv/truedivRealDiv0tf.math.squared_difference/SquaredDifference:z:0'dense8/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.truediv/truediv¥
tf.__operators__.add/AddV2AddV2tf.math.truediv/truediv:z:0tf.math.log/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add/AddV2
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean/Const¡
tf.math.reduce_mean/MeanMeantf.__operators__.add/AddV2:z:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean/Meanß
add_loss/PartitionedCallPartitionedCall!tf.math.reduce_mean/Mean:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_add_loss_layer_call_and_return_conditional_losses_79872
add_loss/PartitionedCall®
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense1_8474*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp°
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const¶
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x¸
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mul®
/dense5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense5_8469*
_output_shapes

:	@*
dtype021
/dense5/kernel/Regularizer/Square/ReadVariableOp°
 dense5/kernel/Regularizer/SquareSquare7dense5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense5/kernel/Regularizer/Square
dense5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense5/kernel/Regularizer/Const¶
dense5/kernel/Regularizer/SumSum$dense5/kernel/Regularizer/Square:y:0(dense5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/Sum
dense5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense5/kernel/Regularizer/mul/x¸
dense5/kernel/Regularizer/mulMul(dense5/kernel/Regularizer/mul/x:output:0&dense5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/mul¯
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense2_8484*
_output_shapes
:	@*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp±
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense2/kernel/Regularizer/Square
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const¶
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x¸
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/mul¯
/dense6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense6_8479*
_output_shapes
:	@*
dtype021
/dense6/kernel/Regularizer/Square/ReadVariableOp±
 dense6/kernel/Regularizer/SquareSquare7dense6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense6/kernel/Regularizer/Square
dense6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense6/kernel/Regularizer/Const¶
dense6/kernel/Regularizer/SumSum$dense6/kernel/Regularizer/Square:y:0(dense6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/Sum
dense6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense6/kernel/Regularizer/mul/x¸
dense6/kernel/Regularizer/mulMul(dense6/kernel/Regularizer/mul/x:output:0&dense6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/mul°
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense3_8494* 
_output_shapes
:
*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp²
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense3/kernel/Regularizer/Square
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const¶
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x¸
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/mul°
/dense7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense7_8489* 
_output_shapes
:
*
dtype021
/dense7/kernel/Regularizer/Square/ReadVariableOp²
 dense7/kernel/Regularizer/SquareSquare7dense7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense7/kernel/Regularizer/Square
dense7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense7/kernel/Regularizer/Const¶
dense7/kernel/Regularizer/SumSum$dense7/kernel/Regularizer/Square:y:0(dense7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/Sum
dense7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense7/kernel/Regularizer/mul/x¸
dense7/kernel/Regularizer/mulMul(dense7/kernel/Regularizer/mul/x:output:0&dense7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/mul
IdentityIdentity'dense4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity'dense8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1o

Identity_2Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 2

Identity_2
NoOpNoOp^dense1/StatefulPartitionedCall0^dense1/kernel/Regularizer/Square/ReadVariableOp^dense2/StatefulPartitionedCall0^dense2/kernel/Regularizer/Square/ReadVariableOp^dense3/StatefulPartitionedCall0^dense3/kernel/Regularizer/Square/ReadVariableOp^dense4/StatefulPartitionedCall^dense5/StatefulPartitionedCall0^dense5/kernel/Regularizer/Square/ReadVariableOp^dense6/StatefulPartitionedCall0^dense6/kernel/Regularizer/Square/ReadVariableOp^dense7/StatefulPartitionedCall0^dense7/kernel/Regularizer/Square/ReadVariableOp^dense8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2b
/dense1/kernel/Regularizer/Square/ReadVariableOp/dense1/kernel/Regularizer/Square/ReadVariableOp2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2b
/dense2/kernel/Regularizer/Square/ReadVariableOp/dense2/kernel/Regularizer/Square/ReadVariableOp2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2b
/dense3/kernel/Regularizer/Square/ReadVariableOp/dense3/kernel/Regularizer/Square/ReadVariableOp2@
dense4/StatefulPartitionedCalldense4/StatefulPartitionedCall2@
dense5/StatefulPartitionedCalldense5/StatefulPartitionedCall2b
/dense5/kernel/Regularizer/Square/ReadVariableOp/dense5/kernel/Regularizer/Square/ReadVariableOp2@
dense6/StatefulPartitionedCalldense6/StatefulPartitionedCall2b
/dense6/kernel/Regularizer/Square/ReadVariableOp/dense6/kernel/Regularizer/Square/ReadVariableOp2@
dense7/StatefulPartitionedCalldense7/StatefulPartitionedCall2b
/dense7/kernel/Regularizer/Square/ReadVariableOp/dense7/kernel/Regularizer/Square/ReadVariableOp2@
dense8/StatefulPartitionedCalldense8/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinput1:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinput2:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinput3
Ò
£
@__inference_dense1_layer_call_and_return_conditional_losses_7845

inputs0
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense1/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÁ
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp°
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const¶
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x¸
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity±
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense1/kernel/Regularizer/Square/ReadVariableOp/dense1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs

ò
@__inference_dense8_layer_call_and_return_conditional_losses_9157

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Softplusq
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð

%__inference_dense3_layer_call_fn_9095

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense3_layer_call_and_return_conditional_losses_79372
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
üm
æ
__inference__traced_save_9434
file_prefix,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop,
(savev2_dense5_kernel_read_readvariableop*
&savev2_dense5_bias_read_readvariableop,
(savev2_dense2_kernel_read_readvariableop*
&savev2_dense2_bias_read_readvariableop,
(savev2_dense6_kernel_read_readvariableop*
&savev2_dense6_bias_read_readvariableop,
(savev2_dense3_kernel_read_readvariableop*
&savev2_dense3_bias_read_readvariableop,
(savev2_dense7_kernel_read_readvariableop*
&savev2_dense7_bias_read_readvariableop,
(savev2_dense4_kernel_read_readvariableop*
&savev2_dense4_bias_read_readvariableop,
(savev2_dense8_kernel_read_readvariableop*
&savev2_dense8_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_dense1_kernel_m_read_readvariableop1
-savev2_adam_dense1_bias_m_read_readvariableop3
/savev2_adam_dense5_kernel_m_read_readvariableop1
-savev2_adam_dense5_bias_m_read_readvariableop3
/savev2_adam_dense2_kernel_m_read_readvariableop1
-savev2_adam_dense2_bias_m_read_readvariableop3
/savev2_adam_dense6_kernel_m_read_readvariableop1
-savev2_adam_dense6_bias_m_read_readvariableop3
/savev2_adam_dense3_kernel_m_read_readvariableop1
-savev2_adam_dense3_bias_m_read_readvariableop3
/savev2_adam_dense7_kernel_m_read_readvariableop1
-savev2_adam_dense7_bias_m_read_readvariableop3
/savev2_adam_dense4_kernel_m_read_readvariableop1
-savev2_adam_dense4_bias_m_read_readvariableop3
/savev2_adam_dense8_kernel_m_read_readvariableop1
-savev2_adam_dense8_bias_m_read_readvariableop3
/savev2_adam_dense1_kernel_v_read_readvariableop1
-savev2_adam_dense1_bias_v_read_readvariableop3
/savev2_adam_dense5_kernel_v_read_readvariableop1
-savev2_adam_dense5_bias_v_read_readvariableop3
/savev2_adam_dense2_kernel_v_read_readvariableop1
-savev2_adam_dense2_bias_v_read_readvariableop3
/savev2_adam_dense6_kernel_v_read_readvariableop1
-savev2_adam_dense6_bias_v_read_readvariableop3
/savev2_adam_dense3_kernel_v_read_readvariableop1
-savev2_adam_dense3_bias_v_read_readvariableop3
/savev2_adam_dense7_kernel_v_read_readvariableop1
-savev2_adam_dense7_bias_v_read_readvariableop3
/savev2_adam_dense4_kernel_v_read_readvariableop1
-savev2_adam_dense4_bias_v_read_readvariableop3
/savev2_adam_dense8_kernel_v_read_readvariableop1
-savev2_adam_dense8_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÂ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*Ô
valueÊBÇ8B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesù
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop(savev2_dense5_kernel_read_readvariableop&savev2_dense5_bias_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableop(savev2_dense6_kernel_read_readvariableop&savev2_dense6_bias_read_readvariableop(savev2_dense3_kernel_read_readvariableop&savev2_dense3_bias_read_readvariableop(savev2_dense7_kernel_read_readvariableop&savev2_dense7_bias_read_readvariableop(savev2_dense4_kernel_read_readvariableop&savev2_dense4_bias_read_readvariableop(savev2_dense8_kernel_read_readvariableop&savev2_dense8_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_dense1_kernel_m_read_readvariableop-savev2_adam_dense1_bias_m_read_readvariableop/savev2_adam_dense5_kernel_m_read_readvariableop-savev2_adam_dense5_bias_m_read_readvariableop/savev2_adam_dense2_kernel_m_read_readvariableop-savev2_adam_dense2_bias_m_read_readvariableop/savev2_adam_dense6_kernel_m_read_readvariableop-savev2_adam_dense6_bias_m_read_readvariableop/savev2_adam_dense3_kernel_m_read_readvariableop-savev2_adam_dense3_bias_m_read_readvariableop/savev2_adam_dense7_kernel_m_read_readvariableop-savev2_adam_dense7_bias_m_read_readvariableop/savev2_adam_dense4_kernel_m_read_readvariableop-savev2_adam_dense4_bias_m_read_readvariableop/savev2_adam_dense8_kernel_m_read_readvariableop-savev2_adam_dense8_bias_m_read_readvariableop/savev2_adam_dense1_kernel_v_read_readvariableop-savev2_adam_dense1_bias_v_read_readvariableop/savev2_adam_dense5_kernel_v_read_readvariableop-savev2_adam_dense5_bias_v_read_readvariableop/savev2_adam_dense2_kernel_v_read_readvariableop-savev2_adam_dense2_bias_v_read_readvariableop/savev2_adam_dense6_kernel_v_read_readvariableop-savev2_adam_dense6_bias_v_read_readvariableop/savev2_adam_dense3_kernel_v_read_readvariableop-savev2_adam_dense3_bias_v_read_readvariableop/savev2_adam_dense7_kernel_v_read_readvariableop-savev2_adam_dense7_bias_v_read_readvariableop/savev2_adam_dense4_kernel_v_read_readvariableop-savev2_adam_dense4_bias_v_read_readvariableop/savev2_adam_dense8_kernel_v_read_readvariableop-savev2_adam_dense8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *F
dtypes<
:28	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*Ë
_input_shapes¹
¶: :	@:@:	@:@:	@::	@::
::
::	::	:: : : : : : : :	@:@:	@:@:	@::	@::
::
::	::	::	@:@:	@:@:	@::	@::
::
::	::	:: 2(
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
:@:$ 

_output_shapes

:	@: 

_output_shapes
:@:%!

_output_shapes
:	@:!

_output_shapes	
::%!

_output_shapes
:	@:!

_output_shapes	
::&	"
 
_output_shapes
:
:!


_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:	@: 

_output_shapes
:@:$ 

_output_shapes

:	@: 

_output_shapes
:@:%!

_output_shapes
:	@:!

_output_shapes	
::%!

_output_shapes
:	@:!

_output_shapes	
::& "
 
_output_shapes
:
:!!

_output_shapes	
::&""
 
_output_shapes
:
:!#

_output_shapes	
::%$!

_output_shapes
:	: %

_output_shapes
::%&!

_output_shapes
:	: '

_output_shapes
::$( 

_output_shapes

:	@: )

_output_shapes
:@:$* 

_output_shapes

:	@: +

_output_shapes
:@:%,!

_output_shapes
:	@:!-

_output_shapes	
::%.!

_output_shapes
:	@:!/

_output_shapes	
::&0"
 
_output_shapes
:
:!1

_output_shapes	
::&2"
 
_output_shapes
:
:!3

_output_shapes	
::%4!

_output_shapes
:	: 5

_output_shapes
::%6!

_output_shapes
:	: 7

_output_shapes
::8

_output_shapes
: 

«
__inference_loss_fn_1_9199J
8dense5_kernel_regularizer_square_readvariableop_resource:	@
identity¢/dense5/kernel/Regularizer/Square/ReadVariableOpÛ
/dense5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense5_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:	@*
dtype021
/dense5/kernel/Regularizer/Square/ReadVariableOp°
 dense5/kernel/Regularizer/SquareSquare7dense5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense5/kernel/Regularizer/Square
dense5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense5/kernel/Regularizer/Const¶
dense5/kernel/Regularizer/SumSum$dense5/kernel/Regularizer/Square:y:0(dense5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/Sum
dense5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense5/kernel/Regularizer/mul/x¸
dense5/kernel/Regularizer/mulMul(dense5/kernel/Regularizer/mul/x:output:0&dense5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/mulk
IdentityIdentity!dense5/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
NoOpNoOp0^dense5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense5/kernel/Regularizer/Square/ReadVariableOp/dense5/kernel/Regularizer/Square/ReadVariableOp

­
__inference_loss_fn_5_9243L
8dense7_kernel_regularizer_square_readvariableop_resource:

identity¢/dense7/kernel/Regularizer/Square/ReadVariableOpÝ
/dense7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense7_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype021
/dense7/kernel/Regularizer/Square/ReadVariableOp²
 dense7/kernel/Regularizer/SquareSquare7dense7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense7/kernel/Regularizer/Square
dense7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense7/kernel/Regularizer/Const¶
dense7/kernel/Regularizer/SumSum$dense7/kernel/Regularizer/Square:y:0(dense7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/Sum
dense7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense7/kernel/Regularizer/mul/x¸
dense7/kernel/Regularizer/mulMul(dense7/kernel/Regularizer/mul/x:output:0&dense7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/mulk
IdentityIdentity!dense7/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
NoOpNoOp0^dense7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense7/kernel/Regularizer/Square/ReadVariableOp/dense7/kernel/Regularizer/Square/ReadVariableOp

Æ
&__inference_model_1_layer_call_fn_8067

input1

input2

input3
unknown:	@
	unknown_0:@
	unknown_1:	@
	unknown_2:@
	unknown_3:	@
	unknown_4:	
	unknown_5:	@
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:

unknown_13:	

unknown_14:
identity

identity_1¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinput1input2input3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_80292
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinput1:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinput2:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinput3
Îr
¬	
A__inference_model_1_layer_call_and_return_conditional_losses_8292

inputs
inputs_1
inputs_2
dense5_8205:	@
dense5_8207:@
dense1_8210:	@
dense1_8212:@
dense6_8215:	@
dense6_8217:	
dense2_8220:	@
dense2_8222:	
dense7_8225:

dense7_8227:	
dense3_8230:

dense3_8232:	
dense8_8235:	
dense8_8237:
dense4_8240:	
dense4_8242:
identity

identity_1

identity_2¢dense1/StatefulPartitionedCall¢/dense1/kernel/Regularizer/Square/ReadVariableOp¢dense2/StatefulPartitionedCall¢/dense2/kernel/Regularizer/Square/ReadVariableOp¢dense3/StatefulPartitionedCall¢/dense3/kernel/Regularizer/Square/ReadVariableOp¢dense4/StatefulPartitionedCall¢dense5/StatefulPartitionedCall¢/dense5/kernel/Regularizer/Square/ReadVariableOp¢dense6/StatefulPartitionedCall¢/dense6/kernel/Regularizer/Square/ReadVariableOp¢dense7/StatefulPartitionedCall¢/dense7/kernel/Regularizer/Square/ReadVariableOp¢dense8/StatefulPartitionedCall
dense5/StatefulPartitionedCallStatefulPartitionedCallinputsdense5_8205dense5_8207*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense5_layer_call_and_return_conditional_losses_78222 
dense5/StatefulPartitionedCall
dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_8210dense1_8212*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_78452 
dense1/StatefulPartitionedCall¦
dense6/StatefulPartitionedCallStatefulPartitionedCall'dense5/StatefulPartitionedCall:output:0dense6_8215dense6_8217*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense6_layer_call_and_return_conditional_losses_78682 
dense6/StatefulPartitionedCall¦
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_8220dense2_8222*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_78912 
dense2/StatefulPartitionedCall¦
dense7/StatefulPartitionedCallStatefulPartitionedCall'dense6/StatefulPartitionedCall:output:0dense7_8225dense7_8227*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense7_layer_call_and_return_conditional_losses_79142 
dense7/StatefulPartitionedCall¦
dense3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0dense3_8230dense3_8232*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense3_layer_call_and_return_conditional_losses_79372 
dense3/StatefulPartitionedCall¥
dense8/StatefulPartitionedCallStatefulPartitionedCall'dense7/StatefulPartitionedCall:output:0dense8_8235dense8_8237*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense8_layer_call_and_return_conditional_losses_79542 
dense8/StatefulPartitionedCall¥
dense4/StatefulPartitionedCallStatefulPartitionedCall'dense3/StatefulPartitionedCall:output:0dense4_8240dense4_8242*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense4_layer_call_and_return_conditional_losses_79702 
dense4/StatefulPartitionedCall
tf.math.log/LogLog'dense8/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.log/LogÖ
,tf.math.squared_difference/SquaredDifferenceSquaredDifference'dense4/StatefulPartitionedCall:output:0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,tf.math.squared_difference/SquaredDifferenceÊ
tf.math.truediv/truedivRealDiv0tf.math.squared_difference/SquaredDifference:z:0'dense8/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.truediv/truediv¥
tf.__operators__.add/AddV2AddV2tf.math.truediv/truediv:z:0tf.math.log/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add/AddV2
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean/Const¡
tf.math.reduce_mean/MeanMeantf.__operators__.add/AddV2:z:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean/Meanß
add_loss/PartitionedCallPartitionedCall!tf.math.reduce_mean/Mean:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_add_loss_layer_call_and_return_conditional_losses_79872
add_loss/PartitionedCall®
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense1_8210*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp°
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const¶
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x¸
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mul®
/dense5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense5_8205*
_output_shapes

:	@*
dtype021
/dense5/kernel/Regularizer/Square/ReadVariableOp°
 dense5/kernel/Regularizer/SquareSquare7dense5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense5/kernel/Regularizer/Square
dense5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense5/kernel/Regularizer/Const¶
dense5/kernel/Regularizer/SumSum$dense5/kernel/Regularizer/Square:y:0(dense5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/Sum
dense5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense5/kernel/Regularizer/mul/x¸
dense5/kernel/Regularizer/mulMul(dense5/kernel/Regularizer/mul/x:output:0&dense5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/mul¯
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense2_8220*
_output_shapes
:	@*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp±
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense2/kernel/Regularizer/Square
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const¶
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x¸
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/mul¯
/dense6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense6_8215*
_output_shapes
:	@*
dtype021
/dense6/kernel/Regularizer/Square/ReadVariableOp±
 dense6/kernel/Regularizer/SquareSquare7dense6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense6/kernel/Regularizer/Square
dense6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense6/kernel/Regularizer/Const¶
dense6/kernel/Regularizer/SumSum$dense6/kernel/Regularizer/Square:y:0(dense6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/Sum
dense6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense6/kernel/Regularizer/mul/x¸
dense6/kernel/Regularizer/mulMul(dense6/kernel/Regularizer/mul/x:output:0&dense6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/mul°
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense3_8230* 
_output_shapes
:
*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp²
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense3/kernel/Regularizer/Square
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const¶
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x¸
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/mul°
/dense7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense7_8225* 
_output_shapes
:
*
dtype021
/dense7/kernel/Regularizer/Square/ReadVariableOp²
 dense7/kernel/Regularizer/SquareSquare7dense7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense7/kernel/Regularizer/Square
dense7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense7/kernel/Regularizer/Const¶
dense7/kernel/Regularizer/SumSum$dense7/kernel/Regularizer/Square:y:0(dense7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/Sum
dense7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense7/kernel/Regularizer/mul/x¸
dense7/kernel/Regularizer/mulMul(dense7/kernel/Regularizer/mul/x:output:0&dense7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/mul
IdentityIdentity'dense4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity'dense8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1o

Identity_2Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 2

Identity_2
NoOpNoOp^dense1/StatefulPartitionedCall0^dense1/kernel/Regularizer/Square/ReadVariableOp^dense2/StatefulPartitionedCall0^dense2/kernel/Regularizer/Square/ReadVariableOp^dense3/StatefulPartitionedCall0^dense3/kernel/Regularizer/Square/ReadVariableOp^dense4/StatefulPartitionedCall^dense5/StatefulPartitionedCall0^dense5/kernel/Regularizer/Square/ReadVariableOp^dense6/StatefulPartitionedCall0^dense6/kernel/Regularizer/Square/ReadVariableOp^dense7/StatefulPartitionedCall0^dense7/kernel/Regularizer/Square/ReadVariableOp^dense8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2b
/dense1/kernel/Regularizer/Square/ReadVariableOp/dense1/kernel/Regularizer/Square/ReadVariableOp2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2b
/dense2/kernel/Regularizer/Square/ReadVariableOp/dense2/kernel/Regularizer/Square/ReadVariableOp2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2b
/dense3/kernel/Regularizer/Square/ReadVariableOp/dense3/kernel/Regularizer/Square/ReadVariableOp2@
dense4/StatefulPartitionedCalldense4/StatefulPartitionedCall2@
dense5/StatefulPartitionedCalldense5/StatefulPartitionedCall2b
/dense5/kernel/Regularizer/Square/ReadVariableOp/dense5/kernel/Regularizer/Square/ReadVariableOp2@
dense6/StatefulPartitionedCalldense6/StatefulPartitionedCall2b
/dense6/kernel/Regularizer/Square/ReadVariableOp/dense6/kernel/Regularizer/Square/ReadVariableOp2@
dense7/StatefulPartitionedCalldense7/StatefulPartitionedCall2b
/dense7/kernel/Regularizer/Square/ReadVariableOp/dense7/kernel/Regularizer/Square/ReadVariableOp2@
dense8/StatefulPartitionedCalldense8/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
¥
@__inference_dense6_layer_call_and_return_conditional_losses_9054

inputs1
matmul_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense6/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ReluÂ
/dense6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype021
/dense6/kernel/Regularizer/Square/ReadVariableOp±
 dense6/kernel/Regularizer/SquareSquare7dense6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense6/kernel/Regularizer/Square
dense6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense6/kernel/Regularizer/Const¶
dense6/kernel/Regularizer/SumSum$dense6/kernel/Regularizer/Square:y:0(dense6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/Sum
dense6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense6/kernel/Regularizer/mul/x¸
dense6/kernel/Regularizer/mulMul(dense6/kernel/Regularizer/mul/x:output:0&dense6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/muln
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity±
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense6/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense6/kernel/Regularizer/Square/ReadVariableOp/dense6/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
äê
ò 
 __inference__traced_restore_9609
file_prefix0
assignvariableop_dense1_kernel:	@,
assignvariableop_1_dense1_bias:@2
 assignvariableop_2_dense5_kernel:	@,
assignvariableop_3_dense5_bias:@3
 assignvariableop_4_dense2_kernel:	@-
assignvariableop_5_dense2_bias:	3
 assignvariableop_6_dense6_kernel:	@-
assignvariableop_7_dense6_bias:	4
 assignvariableop_8_dense3_kernel:
-
assignvariableop_9_dense3_bias:	5
!assignvariableop_10_dense7_kernel:
.
assignvariableop_11_dense7_bias:	4
!assignvariableop_12_dense4_kernel:	-
assignvariableop_13_dense4_bias:4
!assignvariableop_14_dense8_kernel:	-
assignvariableop_15_dense8_bias:'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: #
assignvariableop_21_total: #
assignvariableop_22_count: :
(assignvariableop_23_adam_dense1_kernel_m:	@4
&assignvariableop_24_adam_dense1_bias_m:@:
(assignvariableop_25_adam_dense5_kernel_m:	@4
&assignvariableop_26_adam_dense5_bias_m:@;
(assignvariableop_27_adam_dense2_kernel_m:	@5
&assignvariableop_28_adam_dense2_bias_m:	;
(assignvariableop_29_adam_dense6_kernel_m:	@5
&assignvariableop_30_adam_dense6_bias_m:	<
(assignvariableop_31_adam_dense3_kernel_m:
5
&assignvariableop_32_adam_dense3_bias_m:	<
(assignvariableop_33_adam_dense7_kernel_m:
5
&assignvariableop_34_adam_dense7_bias_m:	;
(assignvariableop_35_adam_dense4_kernel_m:	4
&assignvariableop_36_adam_dense4_bias_m:;
(assignvariableop_37_adam_dense8_kernel_m:	4
&assignvariableop_38_adam_dense8_bias_m::
(assignvariableop_39_adam_dense1_kernel_v:	@4
&assignvariableop_40_adam_dense1_bias_v:@:
(assignvariableop_41_adam_dense5_kernel_v:	@4
&assignvariableop_42_adam_dense5_bias_v:@;
(assignvariableop_43_adam_dense2_kernel_v:	@5
&assignvariableop_44_adam_dense2_bias_v:	;
(assignvariableop_45_adam_dense6_kernel_v:	@5
&assignvariableop_46_adam_dense6_bias_v:	<
(assignvariableop_47_adam_dense3_kernel_v:
5
&assignvariableop_48_adam_dense3_bias_v:	<
(assignvariableop_49_adam_dense7_kernel_v:
5
&assignvariableop_50_adam_dense7_bias_v:	;
(assignvariableop_51_adam_dense4_kernel_v:	4
&assignvariableop_52_adam_dense4_bias_v:;
(assignvariableop_53_adam_dense8_kernel_v:	4
&assignvariableop_54_adam_dense8_bias_v:
identity_56¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9È
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*Ô
valueÊBÇ8B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÿ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÆ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ö
_output_shapesã
à::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¥
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¥
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¥
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense6_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense6_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¥
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9£
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10©
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense7_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11§
AssignVariableOp_11AssignVariableOpassignvariableop_11_dense7_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12©
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense4_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13§
AssignVariableOp_13AssignVariableOpassignvariableop_13_dense4_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14©
AssignVariableOp_14AssignVariableOp!assignvariableop_14_dense8_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15§
AssignVariableOp_15AssignVariableOpassignvariableop_15_dense8_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16¥
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17§
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18§
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¦
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20®
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¡
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¡
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23°
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24®
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_dense1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25°
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_dense5_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26®
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_dense5_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27°
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28®
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_dense2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29°
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense6_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30®
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_dense6_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31°
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_dense3_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32®
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_dense3_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33°
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_dense7_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34®
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_dense7_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35°
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_dense4_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36®
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_dense4_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37°
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_dense8_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38®
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_dense8_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39°
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_dense1_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40®
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_dense1_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41°
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_dense5_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42®
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_dense5_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43°
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_dense2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44®
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_dense2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45°
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_dense6_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46®
AssignVariableOp_46AssignVariableOp&assignvariableop_46_adam_dense6_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47°
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_dense3_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48®
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_dense3_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49°
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_dense7_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50®
AssignVariableOp_50AssignVariableOp&assignvariableop_50_adam_dense7_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51°
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_dense4_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52®
AssignVariableOp_52AssignVariableOp&assignvariableop_52_adam_dense4_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53°
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_dense8_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54®
AssignVariableOp_54AssignVariableOp&assignvariableop_54_adam_dense8_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_549
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_55f
Identity_56IdentityIdentity_55:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_56

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_56Identity_56:output:0*
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Éb
â
__inference__wrapped_model_7794

input1

input2

input3?
-model_1_dense5_matmul_readvariableop_resource:	@<
.model_1_dense5_biasadd_readvariableop_resource:@?
-model_1_dense1_matmul_readvariableop_resource:	@<
.model_1_dense1_biasadd_readvariableop_resource:@@
-model_1_dense6_matmul_readvariableop_resource:	@=
.model_1_dense6_biasadd_readvariableop_resource:	@
-model_1_dense2_matmul_readvariableop_resource:	@=
.model_1_dense2_biasadd_readvariableop_resource:	A
-model_1_dense7_matmul_readvariableop_resource:
=
.model_1_dense7_biasadd_readvariableop_resource:	A
-model_1_dense3_matmul_readvariableop_resource:
=
.model_1_dense3_biasadd_readvariableop_resource:	@
-model_1_dense8_matmul_readvariableop_resource:	<
.model_1_dense8_biasadd_readvariableop_resource:@
-model_1_dense4_matmul_readvariableop_resource:	<
.model_1_dense4_biasadd_readvariableop_resource:
identity

identity_1¢%model_1/dense1/BiasAdd/ReadVariableOp¢$model_1/dense1/MatMul/ReadVariableOp¢%model_1/dense2/BiasAdd/ReadVariableOp¢$model_1/dense2/MatMul/ReadVariableOp¢%model_1/dense3/BiasAdd/ReadVariableOp¢$model_1/dense3/MatMul/ReadVariableOp¢%model_1/dense4/BiasAdd/ReadVariableOp¢$model_1/dense4/MatMul/ReadVariableOp¢%model_1/dense5/BiasAdd/ReadVariableOp¢$model_1/dense5/MatMul/ReadVariableOp¢%model_1/dense6/BiasAdd/ReadVariableOp¢$model_1/dense6/MatMul/ReadVariableOp¢%model_1/dense7/BiasAdd/ReadVariableOp¢$model_1/dense7/MatMul/ReadVariableOp¢%model_1/dense8/BiasAdd/ReadVariableOp¢$model_1/dense8/MatMul/ReadVariableOpº
$model_1/dense5/MatMul/ReadVariableOpReadVariableOp-model_1_dense5_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype02&
$model_1/dense5/MatMul/ReadVariableOp 
model_1/dense5/MatMulMatMulinput1,model_1/dense5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dense5/MatMul¹
%model_1/dense5/BiasAdd/ReadVariableOpReadVariableOp.model_1_dense5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model_1/dense5/BiasAdd/ReadVariableOp½
model_1/dense5/BiasAddBiasAddmodel_1/dense5/MatMul:product:0-model_1/dense5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dense5/BiasAdd
model_1/dense5/ReluRelumodel_1/dense5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dense5/Reluº
$model_1/dense1/MatMul/ReadVariableOpReadVariableOp-model_1_dense1_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype02&
$model_1/dense1/MatMul/ReadVariableOp 
model_1/dense1/MatMulMatMulinput1,model_1/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dense1/MatMul¹
%model_1/dense1/BiasAdd/ReadVariableOpReadVariableOp.model_1_dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model_1/dense1/BiasAdd/ReadVariableOp½
model_1/dense1/BiasAddBiasAddmodel_1/dense1/MatMul:product:0-model_1/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dense1/BiasAdd
model_1/dense1/ReluRelumodel_1/dense1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dense1/Relu»
$model_1/dense6/MatMul/ReadVariableOpReadVariableOp-model_1_dense6_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02&
$model_1/dense6/MatMul/ReadVariableOp¼
model_1/dense6/MatMulMatMul!model_1/dense5/Relu:activations:0,model_1/dense6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense6/MatMulº
%model_1/dense6/BiasAdd/ReadVariableOpReadVariableOp.model_1_dense6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%model_1/dense6/BiasAdd/ReadVariableOp¾
model_1/dense6/BiasAddBiasAddmodel_1/dense6/MatMul:product:0-model_1/dense6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense6/BiasAdd
model_1/dense6/ReluRelumodel_1/dense6/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense6/Relu»
$model_1/dense2/MatMul/ReadVariableOpReadVariableOp-model_1_dense2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02&
$model_1/dense2/MatMul/ReadVariableOp¼
model_1/dense2/MatMulMatMul!model_1/dense1/Relu:activations:0,model_1/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense2/MatMulº
%model_1/dense2/BiasAdd/ReadVariableOpReadVariableOp.model_1_dense2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%model_1/dense2/BiasAdd/ReadVariableOp¾
model_1/dense2/BiasAddBiasAddmodel_1/dense2/MatMul:product:0-model_1/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense2/BiasAdd
model_1/dense2/ReluRelumodel_1/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense2/Relu¼
$model_1/dense7/MatMul/ReadVariableOpReadVariableOp-model_1_dense7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$model_1/dense7/MatMul/ReadVariableOp¼
model_1/dense7/MatMulMatMul!model_1/dense6/Relu:activations:0,model_1/dense7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense7/MatMulº
%model_1/dense7/BiasAdd/ReadVariableOpReadVariableOp.model_1_dense7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%model_1/dense7/BiasAdd/ReadVariableOp¾
model_1/dense7/BiasAddBiasAddmodel_1/dense7/MatMul:product:0-model_1/dense7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense7/BiasAdd
model_1/dense7/ReluRelumodel_1/dense7/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense7/Relu¼
$model_1/dense3/MatMul/ReadVariableOpReadVariableOp-model_1_dense3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$model_1/dense3/MatMul/ReadVariableOp¼
model_1/dense3/MatMulMatMul!model_1/dense2/Relu:activations:0,model_1/dense3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense3/MatMulº
%model_1/dense3/BiasAdd/ReadVariableOpReadVariableOp.model_1_dense3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%model_1/dense3/BiasAdd/ReadVariableOp¾
model_1/dense3/BiasAddBiasAddmodel_1/dense3/MatMul:product:0-model_1/dense3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense3/BiasAdd
model_1/dense3/ReluRelumodel_1/dense3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense3/Relu»
$model_1/dense8/MatMul/ReadVariableOpReadVariableOp-model_1_dense8_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02&
$model_1/dense8/MatMul/ReadVariableOp»
model_1/dense8/MatMulMatMul!model_1/dense7/Relu:activations:0,model_1/dense8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense8/MatMul¹
%model_1/dense8/BiasAdd/ReadVariableOpReadVariableOp.model_1_dense8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model_1/dense8/BiasAdd/ReadVariableOp½
model_1/dense8/BiasAddBiasAddmodel_1/dense8/MatMul:product:0-model_1/dense8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense8/BiasAdd
model_1/dense8/SoftplusSoftplusmodel_1/dense8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense8/Softplus»
$model_1/dense4/MatMul/ReadVariableOpReadVariableOp-model_1_dense4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02&
$model_1/dense4/MatMul/ReadVariableOp»
model_1/dense4/MatMulMatMul!model_1/dense3/Relu:activations:0,model_1/dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense4/MatMul¹
%model_1/dense4/BiasAdd/ReadVariableOpReadVariableOp.model_1_dense4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model_1/dense4/BiasAdd/ReadVariableOp½
model_1/dense4/BiasAddBiasAddmodel_1/dense4/MatMul:product:0-model_1/dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense4/BiasAdd
model_1/tf.math.log/LogLog%model_1/dense8/Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/tf.math.log/LogÜ
4model_1/tf.math.squared_difference/SquaredDifferenceSquaredDifferencemodel_1/dense4/BiasAdd:output:0input2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4model_1/tf.math.squared_difference/SquaredDifferenceà
model_1/tf.math.truediv/truedivRealDiv8model_1/tf.math.squared_difference/SquaredDifference:z:0%model_1/dense8/Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
model_1/tf.math.truediv/truedivÅ
"model_1/tf.__operators__.add/AddV2AddV2#model_1/tf.math.truediv/truediv:z:0model_1/tf.math.log/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"model_1/tf.__operators__.add/AddV2
!model_1/tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!model_1/tf.math.reduce_mean/ConstÁ
 model_1/tf.math.reduce_mean/MeanMean&model_1/tf.__operators__.add/AddV2:z:0*model_1/tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: 2"
 model_1/tf.math.reduce_mean/Meanz
IdentityIdentitymodel_1/dense4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity%model_1/dense8/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1Æ
NoOpNoOp&^model_1/dense1/BiasAdd/ReadVariableOp%^model_1/dense1/MatMul/ReadVariableOp&^model_1/dense2/BiasAdd/ReadVariableOp%^model_1/dense2/MatMul/ReadVariableOp&^model_1/dense3/BiasAdd/ReadVariableOp%^model_1/dense3/MatMul/ReadVariableOp&^model_1/dense4/BiasAdd/ReadVariableOp%^model_1/dense4/MatMul/ReadVariableOp&^model_1/dense5/BiasAdd/ReadVariableOp%^model_1/dense5/MatMul/ReadVariableOp&^model_1/dense6/BiasAdd/ReadVariableOp%^model_1/dense6/MatMul/ReadVariableOp&^model_1/dense7/BiasAdd/ReadVariableOp%^model_1/dense7/MatMul/ReadVariableOp&^model_1/dense8/BiasAdd/ReadVariableOp%^model_1/dense8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2N
%model_1/dense1/BiasAdd/ReadVariableOp%model_1/dense1/BiasAdd/ReadVariableOp2L
$model_1/dense1/MatMul/ReadVariableOp$model_1/dense1/MatMul/ReadVariableOp2N
%model_1/dense2/BiasAdd/ReadVariableOp%model_1/dense2/BiasAdd/ReadVariableOp2L
$model_1/dense2/MatMul/ReadVariableOp$model_1/dense2/MatMul/ReadVariableOp2N
%model_1/dense3/BiasAdd/ReadVariableOp%model_1/dense3/BiasAdd/ReadVariableOp2L
$model_1/dense3/MatMul/ReadVariableOp$model_1/dense3/MatMul/ReadVariableOp2N
%model_1/dense4/BiasAdd/ReadVariableOp%model_1/dense4/BiasAdd/ReadVariableOp2L
$model_1/dense4/MatMul/ReadVariableOp$model_1/dense4/MatMul/ReadVariableOp2N
%model_1/dense5/BiasAdd/ReadVariableOp%model_1/dense5/BiasAdd/ReadVariableOp2L
$model_1/dense5/MatMul/ReadVariableOp$model_1/dense5/MatMul/ReadVariableOp2N
%model_1/dense6/BiasAdd/ReadVariableOp%model_1/dense6/BiasAdd/ReadVariableOp2L
$model_1/dense6/MatMul/ReadVariableOp$model_1/dense6/MatMul/ReadVariableOp2N
%model_1/dense7/BiasAdd/ReadVariableOp%model_1/dense7/BiasAdd/ReadVariableOp2L
$model_1/dense7/MatMul/ReadVariableOp$model_1/dense7/MatMul/ReadVariableOp2N
%model_1/dense8/BiasAdd/ReadVariableOp%model_1/dense8/BiasAdd/ReadVariableOp2L
$model_1/dense8/MatMul/ReadVariableOp$model_1/dense8/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinput1:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinput2:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinput3
ì

%__inference_dense4_layer_call_fn_9146

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense4_layer_call_and_return_conditional_losses_79702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

«
__inference_loss_fn_0_9188J
8dense1_kernel_regularizer_square_readvariableop_resource:	@
identity¢/dense1/kernel/Regularizer/Square/ReadVariableOpÛ
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense1_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp°
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const¶
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x¸
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mulk
IdentityIdentity!dense1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
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

­
__inference_loss_fn_4_9232L
8dense3_kernel_regularizer_square_readvariableop_resource:

identity¢/dense3/kernel/Regularizer/Square/ReadVariableOpÝ
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense3_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp²
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense3/kernel/Regularizer/Square
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const¶
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x¸
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/mulk
IdentityIdentity!dense3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
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

Ì
&__inference_model_1_layer_call_fn_8893
inputs_0
inputs_1
inputs_2
unknown:	@
	unknown_0:@
	unknown_1:	@
	unknown_2:@
	unknown_3:	@
	unknown_4:	
	unknown_5:	@
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:

unknown_13:	

unknown_14:
identity

identity_1¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_80292
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2
ø
Æ
A__inference_model_1_layer_call_and_return_conditional_losses_8851
inputs_0
inputs_1
inputs_27
%dense5_matmul_readvariableop_resource:	@4
&dense5_biasadd_readvariableop_resource:@7
%dense1_matmul_readvariableop_resource:	@4
&dense1_biasadd_readvariableop_resource:@8
%dense6_matmul_readvariableop_resource:	@5
&dense6_biasadd_readvariableop_resource:	8
%dense2_matmul_readvariableop_resource:	@5
&dense2_biasadd_readvariableop_resource:	9
%dense7_matmul_readvariableop_resource:
5
&dense7_biasadd_readvariableop_resource:	9
%dense3_matmul_readvariableop_resource:
5
&dense3_biasadd_readvariableop_resource:	8
%dense8_matmul_readvariableop_resource:	4
&dense8_biasadd_readvariableop_resource:8
%dense4_matmul_readvariableop_resource:	4
&dense4_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢dense1/BiasAdd/ReadVariableOp¢dense1/MatMul/ReadVariableOp¢/dense1/kernel/Regularizer/Square/ReadVariableOp¢dense2/BiasAdd/ReadVariableOp¢dense2/MatMul/ReadVariableOp¢/dense2/kernel/Regularizer/Square/ReadVariableOp¢dense3/BiasAdd/ReadVariableOp¢dense3/MatMul/ReadVariableOp¢/dense3/kernel/Regularizer/Square/ReadVariableOp¢dense4/BiasAdd/ReadVariableOp¢dense4/MatMul/ReadVariableOp¢dense5/BiasAdd/ReadVariableOp¢dense5/MatMul/ReadVariableOp¢/dense5/kernel/Regularizer/Square/ReadVariableOp¢dense6/BiasAdd/ReadVariableOp¢dense6/MatMul/ReadVariableOp¢/dense6/kernel/Regularizer/Square/ReadVariableOp¢dense7/BiasAdd/ReadVariableOp¢dense7/MatMul/ReadVariableOp¢/dense7/kernel/Regularizer/Square/ReadVariableOp¢dense8/BiasAdd/ReadVariableOp¢dense8/MatMul/ReadVariableOp¢
dense5/MatMul/ReadVariableOpReadVariableOp%dense5_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype02
dense5/MatMul/ReadVariableOp
dense5/MatMulMatMulinputs_0$dense5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense5/MatMul¡
dense5/BiasAdd/ReadVariableOpReadVariableOp&dense5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense5/BiasAdd/ReadVariableOp
dense5/BiasAddBiasAdddense5/MatMul:product:0%dense5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense5/BiasAddm
dense5/ReluReludense5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense5/Relu¢
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype02
dense1/MatMul/ReadVariableOp
dense1/MatMulMatMulinputs_0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense1/MatMul¡
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense1/BiasAdd/ReadVariableOp
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense1/BiasAddm
dense1/ReluReludense1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense1/Relu£
dense6/MatMul/ReadVariableOpReadVariableOp%dense6_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense6/MatMul/ReadVariableOp
dense6/MatMulMatMuldense5/Relu:activations:0$dense6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense6/MatMul¢
dense6/BiasAdd/ReadVariableOpReadVariableOp&dense6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense6/BiasAdd/ReadVariableOp
dense6/BiasAddBiasAdddense6/MatMul:product:0%dense6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense6/BiasAddn
dense6/ReluReludense6/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense6/Relu£
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense2/MatMul/ReadVariableOp
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense2/MatMul¢
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense2/BiasAdd/ReadVariableOp
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense2/BiasAddn
dense2/ReluReludense2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense2/Relu¤
dense7/MatMul/ReadVariableOpReadVariableOp%dense7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense7/MatMul/ReadVariableOp
dense7/MatMulMatMuldense6/Relu:activations:0$dense7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense7/MatMul¢
dense7/BiasAdd/ReadVariableOpReadVariableOp&dense7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense7/BiasAdd/ReadVariableOp
dense7/BiasAddBiasAdddense7/MatMul:product:0%dense7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense7/BiasAddn
dense7/ReluReludense7/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense7/Relu¤
dense3/MatMul/ReadVariableOpReadVariableOp%dense3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense3/MatMul/ReadVariableOp
dense3/MatMulMatMuldense2/Relu:activations:0$dense3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense3/MatMul¢
dense3/BiasAdd/ReadVariableOpReadVariableOp&dense3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense3/BiasAdd/ReadVariableOp
dense3/BiasAddBiasAdddense3/MatMul:product:0%dense3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense3/BiasAddn
dense3/ReluReludense3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense3/Relu£
dense8/MatMul/ReadVariableOpReadVariableOp%dense8_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense8/MatMul/ReadVariableOp
dense8/MatMulMatMuldense7/Relu:activations:0$dense8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense8/MatMul¡
dense8/BiasAdd/ReadVariableOpReadVariableOp&dense8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense8/BiasAdd/ReadVariableOp
dense8/BiasAddBiasAdddense8/MatMul:product:0%dense8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense8/BiasAddy
dense8/SoftplusSoftplusdense8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense8/Softplus£
dense4/MatMul/ReadVariableOpReadVariableOp%dense4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense4/MatMul/ReadVariableOp
dense4/MatMulMatMuldense3/Relu:activations:0$dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense4/MatMul¡
dense4/BiasAdd/ReadVariableOpReadVariableOp&dense4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense4/BiasAdd/ReadVariableOp
dense4/BiasAddBiasAdddense4/MatMul:product:0%dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense4/BiasAddz
tf.math.log/LogLogdense8/Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.log/LogÆ
,tf.math.squared_difference/SquaredDifferenceSquaredDifferencedense4/BiasAdd:output:0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,tf.math.squared_difference/SquaredDifferenceÀ
tf.math.truediv/truedivRealDiv0tf.math.squared_difference/SquaredDifference:z:0dense8/Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.truediv/truediv¥
tf.__operators__.add/AddV2AddV2tf.math.truediv/truediv:z:0tf.math.log/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add/AddV2
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean/Const¡
tf.math.reduce_mean/MeanMeantf.__operators__.add/AddV2:z:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean/MeanÈ
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp°
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const¶
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x¸
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mulÈ
/dense5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense5_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype021
/dense5/kernel/Regularizer/Square/ReadVariableOp°
 dense5/kernel/Regularizer/SquareSquare7dense5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense5/kernel/Regularizer/Square
dense5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense5/kernel/Regularizer/Const¶
dense5/kernel/Regularizer/SumSum$dense5/kernel/Regularizer/Square:y:0(dense5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/Sum
dense5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense5/kernel/Regularizer/mul/x¸
dense5/kernel/Regularizer/mulMul(dense5/kernel/Regularizer/mul/x:output:0&dense5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/mulÉ
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp±
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense2/kernel/Regularizer/Square
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const¶
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x¸
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/mulÉ
/dense6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense6_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype021
/dense6/kernel/Regularizer/Square/ReadVariableOp±
 dense6/kernel/Regularizer/SquareSquare7dense6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense6/kernel/Regularizer/Square
dense6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense6/kernel/Regularizer/Const¶
dense6/kernel/Regularizer/SumSum$dense6/kernel/Regularizer/Square:y:0(dense6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/Sum
dense6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense6/kernel/Regularizer/mul/x¸
dense6/kernel/Regularizer/mulMul(dense6/kernel/Regularizer/mul/x:output:0&dense6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/mulÊ
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp²
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense3/kernel/Regularizer/Square
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const¶
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x¸
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/mulÊ
/dense7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype021
/dense7/kernel/Regularizer/Square/ReadVariableOp²
 dense7/kernel/Regularizer/SquareSquare7dense7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense7/kernel/Regularizer/Square
dense7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense7/kernel/Regularizer/Const¶
dense7/kernel/Regularizer/SumSum$dense7/kernel/Regularizer/Square:y:0(dense7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/Sum
dense7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense7/kernel/Regularizer/mul/x¸
dense7/kernel/Regularizer/mulMul(dense7/kernel/Regularizer/mul/x:output:0&dense7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/mulr
IdentityIdentitydense4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity|

Identity_1Identitydense8/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1o

Identity_2Identity!tf.math.reduce_mean/Mean:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_2ò
NoOpNoOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp0^dense1/kernel/Regularizer/Square/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp0^dense2/kernel/Regularizer/Square/ReadVariableOp^dense3/BiasAdd/ReadVariableOp^dense3/MatMul/ReadVariableOp0^dense3/kernel/Regularizer/Square/ReadVariableOp^dense4/BiasAdd/ReadVariableOp^dense4/MatMul/ReadVariableOp^dense5/BiasAdd/ReadVariableOp^dense5/MatMul/ReadVariableOp0^dense5/kernel/Regularizer/Square/ReadVariableOp^dense6/BiasAdd/ReadVariableOp^dense6/MatMul/ReadVariableOp0^dense6/kernel/Regularizer/Square/ReadVariableOp^dense7/BiasAdd/ReadVariableOp^dense7/MatMul/ReadVariableOp0^dense7/kernel/Regularizer/Square/ReadVariableOp^dense8/BiasAdd/ReadVariableOp^dense8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2>
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
dense4/MatMul/ReadVariableOpdense4/MatMul/ReadVariableOp2>
dense5/BiasAdd/ReadVariableOpdense5/BiasAdd/ReadVariableOp2<
dense5/MatMul/ReadVariableOpdense5/MatMul/ReadVariableOp2b
/dense5/kernel/Regularizer/Square/ReadVariableOp/dense5/kernel/Regularizer/Square/ReadVariableOp2>
dense6/BiasAdd/ReadVariableOpdense6/BiasAdd/ReadVariableOp2<
dense6/MatMul/ReadVariableOpdense6/MatMul/ReadVariableOp2b
/dense6/kernel/Regularizer/Square/ReadVariableOp/dense6/kernel/Regularizer/Square/ReadVariableOp2>
dense7/BiasAdd/ReadVariableOpdense7/BiasAdd/ReadVariableOp2<
dense7/MatMul/ReadVariableOpdense7/MatMul/ReadVariableOp2b
/dense7/kernel/Regularizer/Square/ReadVariableOp/dense7/kernel/Regularizer/Square/ReadVariableOp2>
dense8/BiasAdd/ReadVariableOpdense8/BiasAdd/ReadVariableOp2<
dense8/MatMul/ReadVariableOpdense8/MatMul/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2
Í
n
B__inference_add_loss_layer_call_and_return_conditional_losses_7987

inputs
identity

identity_1I
IdentityIdentityinputs*
T0*
_output_shapes
: 2

IdentityM

Identity_1Identityinputs*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
ì

%__inference_dense8_layer_call_fn_9166

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense8_layer_call_and_return_conditional_losses_79542
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
¦
@__inference_dense3_layer_call_and_return_conditional_losses_9086

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense3/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ReluÃ
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp²
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense3/kernel/Regularizer/Square
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const¶
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x¸
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/muln
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity±
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense3/kernel/Regularizer/Square/ReadVariableOp/dense3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
Æ
A__inference_model_1_layer_call_and_return_conditional_losses_8746
inputs_0
inputs_1
inputs_27
%dense5_matmul_readvariableop_resource:	@4
&dense5_biasadd_readvariableop_resource:@7
%dense1_matmul_readvariableop_resource:	@4
&dense1_biasadd_readvariableop_resource:@8
%dense6_matmul_readvariableop_resource:	@5
&dense6_biasadd_readvariableop_resource:	8
%dense2_matmul_readvariableop_resource:	@5
&dense2_biasadd_readvariableop_resource:	9
%dense7_matmul_readvariableop_resource:
5
&dense7_biasadd_readvariableop_resource:	9
%dense3_matmul_readvariableop_resource:
5
&dense3_biasadd_readvariableop_resource:	8
%dense8_matmul_readvariableop_resource:	4
&dense8_biasadd_readvariableop_resource:8
%dense4_matmul_readvariableop_resource:	4
&dense4_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢dense1/BiasAdd/ReadVariableOp¢dense1/MatMul/ReadVariableOp¢/dense1/kernel/Regularizer/Square/ReadVariableOp¢dense2/BiasAdd/ReadVariableOp¢dense2/MatMul/ReadVariableOp¢/dense2/kernel/Regularizer/Square/ReadVariableOp¢dense3/BiasAdd/ReadVariableOp¢dense3/MatMul/ReadVariableOp¢/dense3/kernel/Regularizer/Square/ReadVariableOp¢dense4/BiasAdd/ReadVariableOp¢dense4/MatMul/ReadVariableOp¢dense5/BiasAdd/ReadVariableOp¢dense5/MatMul/ReadVariableOp¢/dense5/kernel/Regularizer/Square/ReadVariableOp¢dense6/BiasAdd/ReadVariableOp¢dense6/MatMul/ReadVariableOp¢/dense6/kernel/Regularizer/Square/ReadVariableOp¢dense7/BiasAdd/ReadVariableOp¢dense7/MatMul/ReadVariableOp¢/dense7/kernel/Regularizer/Square/ReadVariableOp¢dense8/BiasAdd/ReadVariableOp¢dense8/MatMul/ReadVariableOp¢
dense5/MatMul/ReadVariableOpReadVariableOp%dense5_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype02
dense5/MatMul/ReadVariableOp
dense5/MatMulMatMulinputs_0$dense5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense5/MatMul¡
dense5/BiasAdd/ReadVariableOpReadVariableOp&dense5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense5/BiasAdd/ReadVariableOp
dense5/BiasAddBiasAdddense5/MatMul:product:0%dense5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense5/BiasAddm
dense5/ReluReludense5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense5/Relu¢
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype02
dense1/MatMul/ReadVariableOp
dense1/MatMulMatMulinputs_0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense1/MatMul¡
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense1/BiasAdd/ReadVariableOp
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense1/BiasAddm
dense1/ReluReludense1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense1/Relu£
dense6/MatMul/ReadVariableOpReadVariableOp%dense6_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense6/MatMul/ReadVariableOp
dense6/MatMulMatMuldense5/Relu:activations:0$dense6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense6/MatMul¢
dense6/BiasAdd/ReadVariableOpReadVariableOp&dense6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense6/BiasAdd/ReadVariableOp
dense6/BiasAddBiasAdddense6/MatMul:product:0%dense6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense6/BiasAddn
dense6/ReluReludense6/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense6/Relu£
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense2/MatMul/ReadVariableOp
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense2/MatMul¢
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense2/BiasAdd/ReadVariableOp
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense2/BiasAddn
dense2/ReluReludense2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense2/Relu¤
dense7/MatMul/ReadVariableOpReadVariableOp%dense7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense7/MatMul/ReadVariableOp
dense7/MatMulMatMuldense6/Relu:activations:0$dense7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense7/MatMul¢
dense7/BiasAdd/ReadVariableOpReadVariableOp&dense7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense7/BiasAdd/ReadVariableOp
dense7/BiasAddBiasAdddense7/MatMul:product:0%dense7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense7/BiasAddn
dense7/ReluReludense7/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense7/Relu¤
dense3/MatMul/ReadVariableOpReadVariableOp%dense3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense3/MatMul/ReadVariableOp
dense3/MatMulMatMuldense2/Relu:activations:0$dense3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense3/MatMul¢
dense3/BiasAdd/ReadVariableOpReadVariableOp&dense3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense3/BiasAdd/ReadVariableOp
dense3/BiasAddBiasAdddense3/MatMul:product:0%dense3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense3/BiasAddn
dense3/ReluReludense3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense3/Relu£
dense8/MatMul/ReadVariableOpReadVariableOp%dense8_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense8/MatMul/ReadVariableOp
dense8/MatMulMatMuldense7/Relu:activations:0$dense8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense8/MatMul¡
dense8/BiasAdd/ReadVariableOpReadVariableOp&dense8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense8/BiasAdd/ReadVariableOp
dense8/BiasAddBiasAdddense8/MatMul:product:0%dense8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense8/BiasAddy
dense8/SoftplusSoftplusdense8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense8/Softplus£
dense4/MatMul/ReadVariableOpReadVariableOp%dense4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense4/MatMul/ReadVariableOp
dense4/MatMulMatMuldense3/Relu:activations:0$dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense4/MatMul¡
dense4/BiasAdd/ReadVariableOpReadVariableOp&dense4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense4/BiasAdd/ReadVariableOp
dense4/BiasAddBiasAdddense4/MatMul:product:0%dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense4/BiasAddz
tf.math.log/LogLogdense8/Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.log/LogÆ
,tf.math.squared_difference/SquaredDifferenceSquaredDifferencedense4/BiasAdd:output:0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,tf.math.squared_difference/SquaredDifferenceÀ
tf.math.truediv/truedivRealDiv0tf.math.squared_difference/SquaredDifference:z:0dense8/Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.truediv/truediv¥
tf.__operators__.add/AddV2AddV2tf.math.truediv/truediv:z:0tf.math.log/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add/AddV2
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean/Const¡
tf.math.reduce_mean/MeanMeantf.__operators__.add/AddV2:z:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean/MeanÈ
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp°
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const¶
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x¸
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mulÈ
/dense5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense5_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype021
/dense5/kernel/Regularizer/Square/ReadVariableOp°
 dense5/kernel/Regularizer/SquareSquare7dense5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense5/kernel/Regularizer/Square
dense5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense5/kernel/Regularizer/Const¶
dense5/kernel/Regularizer/SumSum$dense5/kernel/Regularizer/Square:y:0(dense5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/Sum
dense5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense5/kernel/Regularizer/mul/x¸
dense5/kernel/Regularizer/mulMul(dense5/kernel/Regularizer/mul/x:output:0&dense5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/mulÉ
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp±
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense2/kernel/Regularizer/Square
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const¶
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x¸
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/mulÉ
/dense6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense6_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype021
/dense6/kernel/Regularizer/Square/ReadVariableOp±
 dense6/kernel/Regularizer/SquareSquare7dense6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense6/kernel/Regularizer/Square
dense6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense6/kernel/Regularizer/Const¶
dense6/kernel/Regularizer/SumSum$dense6/kernel/Regularizer/Square:y:0(dense6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/Sum
dense6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense6/kernel/Regularizer/mul/x¸
dense6/kernel/Regularizer/mulMul(dense6/kernel/Regularizer/mul/x:output:0&dense6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/mulÊ
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp²
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense3/kernel/Regularizer/Square
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const¶
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x¸
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/mulÊ
/dense7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%dense7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype021
/dense7/kernel/Regularizer/Square/ReadVariableOp²
 dense7/kernel/Regularizer/SquareSquare7dense7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense7/kernel/Regularizer/Square
dense7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense7/kernel/Regularizer/Const¶
dense7/kernel/Regularizer/SumSum$dense7/kernel/Regularizer/Square:y:0(dense7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/Sum
dense7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense7/kernel/Regularizer/mul/x¸
dense7/kernel/Regularizer/mulMul(dense7/kernel/Regularizer/mul/x:output:0&dense7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/mulr
IdentityIdentitydense4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity|

Identity_1Identitydense8/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1o

Identity_2Identity!tf.math.reduce_mean/Mean:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_2ò
NoOpNoOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp0^dense1/kernel/Regularizer/Square/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp0^dense2/kernel/Regularizer/Square/ReadVariableOp^dense3/BiasAdd/ReadVariableOp^dense3/MatMul/ReadVariableOp0^dense3/kernel/Regularizer/Square/ReadVariableOp^dense4/BiasAdd/ReadVariableOp^dense4/MatMul/ReadVariableOp^dense5/BiasAdd/ReadVariableOp^dense5/MatMul/ReadVariableOp0^dense5/kernel/Regularizer/Square/ReadVariableOp^dense6/BiasAdd/ReadVariableOp^dense6/MatMul/ReadVariableOp0^dense6/kernel/Regularizer/Square/ReadVariableOp^dense7/BiasAdd/ReadVariableOp^dense7/MatMul/ReadVariableOp0^dense7/kernel/Regularizer/Square/ReadVariableOp^dense8/BiasAdd/ReadVariableOp^dense8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2>
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
dense4/MatMul/ReadVariableOpdense4/MatMul/ReadVariableOp2>
dense5/BiasAdd/ReadVariableOpdense5/BiasAdd/ReadVariableOp2<
dense5/MatMul/ReadVariableOpdense5/MatMul/ReadVariableOp2b
/dense5/kernel/Regularizer/Square/ReadVariableOp/dense5/kernel/Regularizer/Square/ReadVariableOp2>
dense6/BiasAdd/ReadVariableOpdense6/BiasAdd/ReadVariableOp2<
dense6/MatMul/ReadVariableOpdense6/MatMul/ReadVariableOp2b
/dense6/kernel/Regularizer/Square/ReadVariableOp/dense6/kernel/Regularizer/Square/ReadVariableOp2>
dense7/BiasAdd/ReadVariableOpdense7/BiasAdd/ReadVariableOp2<
dense7/MatMul/ReadVariableOpdense7/MatMul/ReadVariableOp2b
/dense7/kernel/Regularizer/Square/ReadVariableOp/dense7/kernel/Regularizer/Square/ReadVariableOp2>
dense8/BiasAdd/ReadVariableOpdense8/BiasAdd/ReadVariableOp2<
dense8/MatMul/ReadVariableOpdense8/MatMul/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2
Ü
¥
@__inference_dense2_layer_call_and_return_conditional_losses_9022

inputs1
matmul_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense2/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ReluÂ
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp±
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense2/kernel/Regularizer/Square
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const¶
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x¸
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/muln
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity±
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense2/kernel/Regularizer/Square/ReadVariableOp/dense2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Îr
¬	
A__inference_model_1_layer_call_and_return_conditional_losses_8029

inputs
inputs_1
inputs_2
dense5_7823:	@
dense5_7825:@
dense1_7846:	@
dense1_7848:@
dense6_7869:	@
dense6_7871:	
dense2_7892:	@
dense2_7894:	
dense7_7915:

dense7_7917:	
dense3_7938:

dense3_7940:	
dense8_7955:	
dense8_7957:
dense4_7971:	
dense4_7973:
identity

identity_1

identity_2¢dense1/StatefulPartitionedCall¢/dense1/kernel/Regularizer/Square/ReadVariableOp¢dense2/StatefulPartitionedCall¢/dense2/kernel/Regularizer/Square/ReadVariableOp¢dense3/StatefulPartitionedCall¢/dense3/kernel/Regularizer/Square/ReadVariableOp¢dense4/StatefulPartitionedCall¢dense5/StatefulPartitionedCall¢/dense5/kernel/Regularizer/Square/ReadVariableOp¢dense6/StatefulPartitionedCall¢/dense6/kernel/Regularizer/Square/ReadVariableOp¢dense7/StatefulPartitionedCall¢/dense7/kernel/Regularizer/Square/ReadVariableOp¢dense8/StatefulPartitionedCall
dense5/StatefulPartitionedCallStatefulPartitionedCallinputsdense5_7823dense5_7825*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense5_layer_call_and_return_conditional_losses_78222 
dense5/StatefulPartitionedCall
dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_7846dense1_7848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_78452 
dense1/StatefulPartitionedCall¦
dense6/StatefulPartitionedCallStatefulPartitionedCall'dense5/StatefulPartitionedCall:output:0dense6_7869dense6_7871*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense6_layer_call_and_return_conditional_losses_78682 
dense6/StatefulPartitionedCall¦
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_7892dense2_7894*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_78912 
dense2/StatefulPartitionedCall¦
dense7/StatefulPartitionedCallStatefulPartitionedCall'dense6/StatefulPartitionedCall:output:0dense7_7915dense7_7917*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense7_layer_call_and_return_conditional_losses_79142 
dense7/StatefulPartitionedCall¦
dense3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0dense3_7938dense3_7940*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense3_layer_call_and_return_conditional_losses_79372 
dense3/StatefulPartitionedCall¥
dense8/StatefulPartitionedCallStatefulPartitionedCall'dense7/StatefulPartitionedCall:output:0dense8_7955dense8_7957*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense8_layer_call_and_return_conditional_losses_79542 
dense8/StatefulPartitionedCall¥
dense4/StatefulPartitionedCallStatefulPartitionedCall'dense3/StatefulPartitionedCall:output:0dense4_7971dense4_7973*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense4_layer_call_and_return_conditional_losses_79702 
dense4/StatefulPartitionedCall
tf.math.log/LogLog'dense8/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.log/LogÖ
,tf.math.squared_difference/SquaredDifferenceSquaredDifference'dense4/StatefulPartitionedCall:output:0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,tf.math.squared_difference/SquaredDifferenceÊ
tf.math.truediv/truedivRealDiv0tf.math.squared_difference/SquaredDifference:z:0'dense8/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.truediv/truediv¥
tf.__operators__.add/AddV2AddV2tf.math.truediv/truediv:z:0tf.math.log/Log:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add/AddV2
tf.math.reduce_mean/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
tf.math.reduce_mean/Const¡
tf.math.reduce_mean/MeanMeantf.__operators__.add/AddV2:z:0"tf.math.reduce_mean/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean/Meanß
add_loss/PartitionedCallPartitionedCall!tf.math.reduce_mean/Mean:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_add_loss_layer_call_and_return_conditional_losses_79872
add_loss/PartitionedCall®
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense1_7846*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp°
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const¶
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x¸
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mul®
/dense5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense5_7823*
_output_shapes

:	@*
dtype021
/dense5/kernel/Regularizer/Square/ReadVariableOp°
 dense5/kernel/Regularizer/SquareSquare7dense5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense5/kernel/Regularizer/Square
dense5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense5/kernel/Regularizer/Const¶
dense5/kernel/Regularizer/SumSum$dense5/kernel/Regularizer/Square:y:0(dense5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/Sum
dense5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense5/kernel/Regularizer/mul/x¸
dense5/kernel/Regularizer/mulMul(dense5/kernel/Regularizer/mul/x:output:0&dense5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/mul¯
/dense2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense2_7892*
_output_shapes
:	@*
dtype021
/dense2/kernel/Regularizer/Square/ReadVariableOp±
 dense2/kernel/Regularizer/SquareSquare7dense2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense2/kernel/Regularizer/Square
dense2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense2/kernel/Regularizer/Const¶
dense2/kernel/Regularizer/SumSum$dense2/kernel/Regularizer/Square:y:0(dense2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/Sum
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense2/kernel/Regularizer/mul/x¸
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0&dense2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense2/kernel/Regularizer/mul¯
/dense6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense6_7869*
_output_shapes
:	@*
dtype021
/dense6/kernel/Regularizer/Square/ReadVariableOp±
 dense6/kernel/Regularizer/SquareSquare7dense6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 dense6/kernel/Regularizer/Square
dense6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense6/kernel/Regularizer/Const¶
dense6/kernel/Regularizer/SumSum$dense6/kernel/Regularizer/Square:y:0(dense6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/Sum
dense6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense6/kernel/Regularizer/mul/x¸
dense6/kernel/Regularizer/mulMul(dense6/kernel/Regularizer/mul/x:output:0&dense6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense6/kernel/Regularizer/mul°
/dense3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense3_7938* 
_output_shapes
:
*
dtype021
/dense3/kernel/Regularizer/Square/ReadVariableOp²
 dense3/kernel/Regularizer/SquareSquare7dense3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense3/kernel/Regularizer/Square
dense3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense3/kernel/Regularizer/Const¶
dense3/kernel/Regularizer/SumSum$dense3/kernel/Regularizer/Square:y:0(dense3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/Sum
dense3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense3/kernel/Regularizer/mul/x¸
dense3/kernel/Regularizer/mulMul(dense3/kernel/Regularizer/mul/x:output:0&dense3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense3/kernel/Regularizer/mul°
/dense7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense7_7915* 
_output_shapes
:
*
dtype021
/dense7/kernel/Regularizer/Square/ReadVariableOp²
 dense7/kernel/Regularizer/SquareSquare7dense7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense7/kernel/Regularizer/Square
dense7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense7/kernel/Regularizer/Const¶
dense7/kernel/Regularizer/SumSum$dense7/kernel/Regularizer/Square:y:0(dense7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/Sum
dense7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense7/kernel/Regularizer/mul/x¸
dense7/kernel/Regularizer/mulMul(dense7/kernel/Regularizer/mul/x:output:0&dense7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/mul
IdentityIdentity'dense4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity'dense8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1o

Identity_2Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 2

Identity_2
NoOpNoOp^dense1/StatefulPartitionedCall0^dense1/kernel/Regularizer/Square/ReadVariableOp^dense2/StatefulPartitionedCall0^dense2/kernel/Regularizer/Square/ReadVariableOp^dense3/StatefulPartitionedCall0^dense3/kernel/Regularizer/Square/ReadVariableOp^dense4/StatefulPartitionedCall^dense5/StatefulPartitionedCall0^dense5/kernel/Regularizer/Square/ReadVariableOp^dense6/StatefulPartitionedCall0^dense6/kernel/Regularizer/Square/ReadVariableOp^dense7/StatefulPartitionedCall0^dense7/kernel/Regularizer/Square/ReadVariableOp^dense8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2b
/dense1/kernel/Regularizer/Square/ReadVariableOp/dense1/kernel/Regularizer/Square/ReadVariableOp2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2b
/dense2/kernel/Regularizer/Square/ReadVariableOp/dense2/kernel/Regularizer/Square/ReadVariableOp2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2b
/dense3/kernel/Regularizer/Square/ReadVariableOp/dense3/kernel/Regularizer/Square/ReadVariableOp2@
dense4/StatefulPartitionedCalldense4/StatefulPartitionedCall2@
dense5/StatefulPartitionedCalldense5/StatefulPartitionedCall2b
/dense5/kernel/Regularizer/Square/ReadVariableOp/dense5/kernel/Regularizer/Square/ReadVariableOp2@
dense6/StatefulPartitionedCalldense6/StatefulPartitionedCall2b
/dense6/kernel/Regularizer/Square/ReadVariableOp/dense6/kernel/Regularizer/Square/ReadVariableOp2@
dense7/StatefulPartitionedCalldense7/StatefulPartitionedCall2b
/dense7/kernel/Regularizer/Square/ReadVariableOp/dense7/kernel/Regularizer/Square/ReadVariableOp2@
dense8/StatefulPartitionedCalldense8/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
£
@__inference_dense1_layer_call_and_return_conditional_losses_8958

inputs0
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense1/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÁ
/dense1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	@*
dtype021
/dense1/kernel/Regularizer/Square/ReadVariableOp°
 dense1/kernel/Regularizer/SquareSquare7dense1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense1/kernel/Regularizer/Square
dense1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense1/kernel/Regularizer/Const¶
dense1/kernel/Regularizer/SumSum$dense1/kernel/Regularizer/Square:y:0(dense1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/Sum
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense1/kernel/Regularizer/mul/x¸
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0&dense1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense1/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity±
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense1/kernel/Regularizer/Square/ReadVariableOp/dense1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs

ò
@__inference_dense8_layer_call_and_return_conditional_losses_7954

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Softplusq
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

ò
@__inference_dense4_layer_call_and_return_conditional_losses_9137

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
¦
@__inference_dense7_layer_call_and_return_conditional_losses_9118

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense7/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ReluÃ
/dense7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype021
/dense7/kernel/Regularizer/Square/ReadVariableOp²
 dense7/kernel/Regularizer/SquareSquare7dense7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense7/kernel/Regularizer/Square
dense7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense7/kernel/Regularizer/Const¶
dense7/kernel/Regularizer/SumSum$dense7/kernel/Regularizer/Square:y:0(dense7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/Sum
dense7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense7/kernel/Regularizer/mul/x¸
dense7/kernel/Regularizer/mulMul(dense7/kernel/Regularizer/mul/x:output:0&dense7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/muln
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity±
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense7/kernel/Regularizer/Square/ReadVariableOp/dense7/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
¦
@__inference_dense7_layer_call_and_return_conditional_losses_7914

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense7/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ReluÃ
/dense7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype021
/dense7/kernel/Regularizer/Square/ReadVariableOp²
 dense7/kernel/Regularizer/SquareSquare7dense7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2"
 dense7/kernel/Regularizer/Square
dense7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense7/kernel/Regularizer/Const¶
dense7/kernel/Regularizer/SumSum$dense7/kernel/Regularizer/Square:y:0(dense7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/Sum
dense7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense7/kernel/Regularizer/mul/x¸
dense7/kernel/Regularizer/mulMul(dense7/kernel/Regularizer/mul/x:output:0&dense7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense7/kernel/Regularizer/muln
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity±
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense7/kernel/Regularizer/Square/ReadVariableOp/dense7/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð

%__inference_dense7_layer_call_fn_9127

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense7_layer_call_and_return_conditional_losses_79142
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

ò
@__inference_dense4_layer_call_and_return_conditional_losses_7970

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Æ
&__inference_model_1_layer_call_fn_8372

input1

input2

input3
unknown:	@
	unknown_0:@
	unknown_1:	@
	unknown_2:@
	unknown_3:	@
	unknown_4:	
	unknown_5:	@
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:

unknown_13:	

unknown_14:
identity

identity_1¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinput1input2input3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_82922
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinput1:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinput2:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinput3
í

%__inference_dense6_layer_call_fn_9063

inputs
unknown:	@
	unknown_0:	
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense6_layer_call_and_return_conditional_losses_78682
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ò
£
@__inference_dense5_layer_call_and_return_conditional_losses_7822

inputs0
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense5/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÁ
/dense5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	@*
dtype021
/dense5/kernel/Regularizer/Square/ReadVariableOp°
 dense5/kernel/Regularizer/SquareSquare7dense5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	@2"
 dense5/kernel/Regularizer/Square
dense5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense5/kernel/Regularizer/Const¶
dense5/kernel/Regularizer/SumSum$dense5/kernel/Regularizer/Square:y:0(dense5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/Sum
dense5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense5/kernel/Regularizer/mul/x¸
dense5/kernel/Regularizer/mulMul(dense5/kernel/Regularizer/mul/x:output:0&dense5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense5/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity±
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense5/kernel/Regularizer/Square/ReadVariableOp/dense5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
÷
C
'__inference_add_loss_layer_call_fn_9177

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_add_loss_layer_call_and_return_conditional_losses_79872
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
é

%__inference_dense5_layer_call_fn_8999

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense5_layer_call_and_return_conditional_losses_78222
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ù
serving_defaultÅ
9
input1/
serving_default_input1:0ÿÿÿÿÿÿÿÿÿ	
9
input2/
serving_default_input2:0ÿÿÿÿÿÿÿÿÿ
9
input3/
serving_default_input3:0ÿÿÿÿÿÿÿÿÿ:
dense40
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ:
dense80
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ù»
¤
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
	optimizer
loss
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+®&call_and_return_all_conditional_losses
¯__call__
°_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
½

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+±&call_and_return_all_conditional_losses
²__call__"
_tf_keras_layer
½

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
+³&call_and_return_all_conditional_losses
´__call__"
_tf_keras_layer
½

%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
+µ&call_and_return_all_conditional_losses
¶__call__"
_tf_keras_layer
½

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
+·&call_and_return_all_conditional_losses
¸__call__"
_tf_keras_layer
½

1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+¹&call_and_return_all_conditional_losses
º__call__"
_tf_keras_layer
½

7kernel
8bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
+»&call_and_return_all_conditional_losses
¼__call__"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
½

=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
+½&call_and_return_all_conditional_losses
¾__call__"
_tf_keras_layer
½

Ckernel
Dbias
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
+¿&call_and_return_all_conditional_losses
À__call__"
_tf_keras_layer
(
I	keras_api"
_tf_keras_layer
(
J	keras_api"
_tf_keras_layer
(
K	keras_api"
_tf_keras_layer
(
L	keras_api"
_tf_keras_layer
(
M	keras_api"
_tf_keras_layer
§
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
+Á&call_and_return_all_conditional_losses
Â__call__"
_tf_keras_layer

Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratemmm m%m&m+m,m1m2m7m8m=m>mCmDmvvv  v¡%v¢&v£+v¤,v¥1v¦2v§7v¨8v©=vª>v«Cv¬Dv­"
	optimizer
 "
trackable_dict_wrapper
P
Ã0
Ä1
Å2
Æ3
Ç4
È5"
trackable_list_wrapper

0
1
2
 3
%4
&5
+6
,7
18
29
710
811
=12
>13
C14
D15"
trackable_list_wrapper

0
1
2
 3
%4
&5
+6
,7
18
29
710
811
=12
>13
C14
D15"
trackable_list_wrapper
Î
regularization_losses
	variables
Wnon_trainable_variables
trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
¯__call__
°_default_save_signature
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
-
Éserving_default"
signature_map
:	@2dense1/kernel
:@2dense1/bias
(
Ã0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
regularization_losses
	variables
\non_trainable_variables
trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
²__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
:	@2dense5/kernel
:@2dense5/bias
(
Ä0"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
°
!regularization_losses
"	variables
anon_trainable_variables
#trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
´__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
 :	@2dense2/kernel
:2dense2/bias
(
Å0"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
°
'regularization_losses
(	variables
fnon_trainable_variables
)trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
¶__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
 :	@2dense6/kernel
:2dense6/bias
(
Æ0"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
°
-regularization_losses
.	variables
knon_trainable_variables
/trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
¸__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
!:
2dense3/kernel
:2dense3/bias
(
Ç0"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
°
3regularization_losses
4	variables
pnon_trainable_variables
5trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
º__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
!:
2dense7/kernel
:2dense7/bias
(
È0"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
°
9regularization_losses
:	variables
unon_trainable_variables
;trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
¼__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 :	2dense4/kernel
:2dense4/bias
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
°
?regularization_losses
@	variables
znon_trainable_variables
Atrainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
¾__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 :	2dense8/kernel
:2dense8/bias
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
´
Eregularization_losses
F	variables
non_trainable_variables
Gtrainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
À__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Nregularization_losses
O	variables
non_trainable_variables
Ptrainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Â__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
(
0"
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
(
Ã0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Ä0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Å0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Æ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Ç0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
È0"
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
R

total

count
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
$:"	@2Adam/dense1/kernel/m
:@2Adam/dense1/bias/m
$:"	@2Adam/dense5/kernel/m
:@2Adam/dense5/bias/m
%:#	@2Adam/dense2/kernel/m
:2Adam/dense2/bias/m
%:#	@2Adam/dense6/kernel/m
:2Adam/dense6/bias/m
&:$
2Adam/dense3/kernel/m
:2Adam/dense3/bias/m
&:$
2Adam/dense7/kernel/m
:2Adam/dense7/bias/m
%:#	2Adam/dense4/kernel/m
:2Adam/dense4/bias/m
%:#	2Adam/dense8/kernel/m
:2Adam/dense8/bias/m
$:"	@2Adam/dense1/kernel/v
:@2Adam/dense1/bias/v
$:"	@2Adam/dense5/kernel/v
:@2Adam/dense5/bias/v
%:#	@2Adam/dense2/kernel/v
:2Adam/dense2/bias/v
%:#	@2Adam/dense6/kernel/v
:2Adam/dense6/bias/v
&:$
2Adam/dense3/kernel/v
:2Adam/dense3/bias/v
&:$
2Adam/dense7/kernel/v
:2Adam/dense7/bias/v
%:#	2Adam/dense4/kernel/v
:2Adam/dense4/bias/v
%:#	2Adam/dense8/kernel/v
:2Adam/dense8/bias/v
Ò2Ï
A__inference_model_1_layer_call_and_return_conditional_losses_8746
A__inference_model_1_layer_call_and_return_conditional_losses_8851
A__inference_model_1_layer_call_and_return_conditional_losses_8464
A__inference_model_1_layer_call_and_return_conditional_losses_8556À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
&__inference_model_1_layer_call_fn_8067
&__inference_model_1_layer_call_fn_8893
&__inference_model_1_layer_call_fn_8935
&__inference_model_1_layer_call_fn_8372À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÙBÖ
__inference__wrapped_model_7794input1input2input3"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense1_layer_call_and_return_conditional_losses_8958¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_dense1_layer_call_fn_8967¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense5_layer_call_and_return_conditional_losses_8990¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_dense5_layer_call_fn_8999¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense2_layer_call_and_return_conditional_losses_9022¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_dense2_layer_call_fn_9031¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense6_layer_call_and_return_conditional_losses_9054¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_dense6_layer_call_fn_9063¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense3_layer_call_and_return_conditional_losses_9086¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_dense3_layer_call_fn_9095¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense7_layer_call_and_return_conditional_losses_9118¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_dense7_layer_call_fn_9127¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense4_layer_call_and_return_conditional_losses_9137¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_dense4_layer_call_fn_9146¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense8_layer_call_and_return_conditional_losses_9157¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_dense8_layer_call_fn_9166¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_add_loss_layer_call_and_return_conditional_losses_9171¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_add_loss_layer_call_fn_9177¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
±2®
__inference_loss_fn_0_9188
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
±2®
__inference_loss_fn_1_9199
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
±2®
__inference_loss_fn_2_9210
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
±2®
__inference_loss_fn_3_9221
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
±2®
__inference_loss_fn_4_9232
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
±2®
__inference_loss_fn_5_9243
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
ÖBÓ
"__inference_signature_wrapper_8641input1input2input3"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
__inference__wrapped_model_7794é +,%&7812CD=>x¢u
n¢k
if
 
input1ÿÿÿÿÿÿÿÿÿ	
 
input2ÿÿÿÿÿÿÿÿÿ
 
input3ÿÿÿÿÿÿÿÿÿ
ª "[ªX
*
dense4 
dense4ÿÿÿÿÿÿÿÿÿ
*
dense8 
dense8ÿÿÿÿÿÿÿÿÿ
B__inference_add_loss_layer_call_and_return_conditional_losses_9171D¢
¢

inputs 
ª ""¢


0 

	
1/0 T
'__inference_add_loss_layer_call_fn_9177)¢
¢

inputs 
ª "  
@__inference_dense1_layer_call_and_return_conditional_losses_8958\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ	
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 x
%__inference_dense1_layer_call_fn_8967O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ	
ª "ÿÿÿÿÿÿÿÿÿ@¡
@__inference_dense2_layer_call_and_return_conditional_losses_9022]%&/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 y
%__inference_dense2_layer_call_fn_9031P%&/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¢
@__inference_dense3_layer_call_and_return_conditional_losses_9086^120¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 z
%__inference_dense3_layer_call_fn_9095Q120¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¡
@__inference_dense4_layer_call_and_return_conditional_losses_9137]=>0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
%__inference_dense4_layer_call_fn_9146P=>0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ 
@__inference_dense5_layer_call_and_return_conditional_losses_8990\ /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ	
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 x
%__inference_dense5_layer_call_fn_8999O /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ	
ª "ÿÿÿÿÿÿÿÿÿ@¡
@__inference_dense6_layer_call_and_return_conditional_losses_9054]+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 y
%__inference_dense6_layer_call_fn_9063P+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¢
@__inference_dense7_layer_call_and_return_conditional_losses_9118^780¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 z
%__inference_dense7_layer_call_fn_9127Q780¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¡
@__inference_dense8_layer_call_and_return_conditional_losses_9157]CD0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
%__inference_dense8_layer_call_fn_9166PCD0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ9
__inference_loss_fn_0_9188¢

¢ 
ª " 9
__inference_loss_fn_1_9199¢

¢ 
ª " 9
__inference_loss_fn_2_9210%¢

¢ 
ª " 9
__inference_loss_fn_3_9221+¢

¢ 
ª " 9
__inference_loss_fn_4_92321¢

¢ 
ª " 9
__inference_loss_fn_5_92437¢

¢ 
ª " ¶
A__inference_model_1_layer_call_and_return_conditional_losses_8464ð +,%&7812CD=>¢}
v¢s
if
 
input1ÿÿÿÿÿÿÿÿÿ	
 
input2ÿÿÿÿÿÿÿÿÿ
 
input3ÿÿÿÿÿÿÿÿÿ
p 

 
ª "Y¢V
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

	
1/0 ¶
A__inference_model_1_layer_call_and_return_conditional_losses_8556ð +,%&7812CD=>¢}
v¢s
if
 
input1ÿÿÿÿÿÿÿÿÿ	
 
input2ÿÿÿÿÿÿÿÿÿ
 
input3ÿÿÿÿÿÿÿÿÿ
p

 
ª "Y¢V
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

	
1/0 ½
A__inference_model_1_layer_call_and_return_conditional_losses_8746÷ +,%&7812CD=>¢
|¢y
ol
"
inputs/0ÿÿÿÿÿÿÿÿÿ	
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "Y¢V
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

	
1/0 ½
A__inference_model_1_layer_call_and_return_conditional_losses_8851÷ +,%&7812CD=>¢
|¢y
ol
"
inputs/0ÿÿÿÿÿÿÿÿÿ	
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
p

 
ª "Y¢V
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

	
1/0 ÿ
&__inference_model_1_layer_call_fn_8067Ô +,%&7812CD=>¢}
v¢s
if
 
input1ÿÿÿÿÿÿÿÿÿ	
 
input2ÿÿÿÿÿÿÿÿÿ
 
input3ÿÿÿÿÿÿÿÿÿ
p 

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿÿ
&__inference_model_1_layer_call_fn_8372Ô +,%&7812CD=>¢}
v¢s
if
 
input1ÿÿÿÿÿÿÿÿÿ	
 
input2ÿÿÿÿÿÿÿÿÿ
 
input3ÿÿÿÿÿÿÿÿÿ
p

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ
&__inference_model_1_layer_call_fn_8893Û +,%&7812CD=>¢
|¢y
ol
"
inputs/0ÿÿÿÿÿÿÿÿÿ	
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ
&__inference_model_1_layer_call_fn_8935Û +,%&7812CD=>¢
|¢y
ol
"
inputs/0ÿÿÿÿÿÿÿÿÿ	
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
p

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ­
"__inference_signature_wrapper_8641 +,%&7812CD=>¢
¢ 
ª
*
input1 
input1ÿÿÿÿÿÿÿÿÿ	
*
input2 
input2ÿÿÿÿÿÿÿÿÿ
*
input3 
input3ÿÿÿÿÿÿÿÿÿ"[ªX
*
dense4 
dense4ÿÿÿÿÿÿÿÿÿ
*
dense8 
dense8ÿÿÿÿÿÿÿÿÿ