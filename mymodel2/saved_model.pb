“µ
Щэ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
Њ
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.0.02unknown8њх
x
dense_9/kernelVarHandleOp*
shape
:*
shared_namedense_9/kernel*
dtype0*
_output_shapes
: 
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
dtype0*
_output_shapes

:
p
dense_9/biasVarHandleOp*
shape:*
shared_namedense_9/bias*
dtype0*
_output_shapes
: 
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
dtype0*
_output_shapes
:
f
	Adam/iterVarHandleOp*
shape: *
shared_name	Adam/iter*
dtype0	*
_output_shapes
: 
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
shape: *
shared_nameAdam/beta_1*
dtype0*
_output_shapes
: 
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
shape: *
shared_nameAdam/beta_2*
dtype0*
_output_shapes
: 
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
shape: *
shared_name
Adam/decay*
dtype0*
_output_shapes
: 
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*
shape: *#
shared_nameAdam/learning_rate*
dtype0*
_output_shapes
: 
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
x
lstm_18/kernelVarHandleOp*
shape
:P*
shared_namelstm_18/kernel*
dtype0*
_output_shapes
: 
q
"lstm_18/kernel/Read/ReadVariableOpReadVariableOplstm_18/kernel*
dtype0*
_output_shapes

:P
М
lstm_18/recurrent_kernelVarHandleOp*
shape
:P*)
shared_namelstm_18/recurrent_kernel*
dtype0*
_output_shapes
: 
Е
,lstm_18/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm_18/recurrent_kernel*
dtype0*
_output_shapes

:P
p
lstm_18/biasVarHandleOp*
shape:P*
shared_namelstm_18/bias*
dtype0*
_output_shapes
: 
i
 lstm_18/bias/Read/ReadVariableOpReadVariableOplstm_18/bias*
dtype0*
_output_shapes
:P
x
lstm_19/kernelVarHandleOp*
shape
:P*
shared_namelstm_19/kernel*
dtype0*
_output_shapes
: 
q
"lstm_19/kernel/Read/ReadVariableOpReadVariableOplstm_19/kernel*
dtype0*
_output_shapes

:P
М
lstm_19/recurrent_kernelVarHandleOp*
shape
:P*)
shared_namelstm_19/recurrent_kernel*
dtype0*
_output_shapes
: 
Е
,lstm_19/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm_19/recurrent_kernel*
dtype0*
_output_shapes

:P
p
lstm_19/biasVarHandleOp*
shape:P*
shared_namelstm_19/bias*
dtype0*
_output_shapes
: 
i
 lstm_19/bias/Read/ReadVariableOpReadVariableOplstm_19/bias*
dtype0*
_output_shapes
:P
Ж
Adam/dense_9/kernel/mVarHandleOp*
shape
:*&
shared_nameAdam/dense_9/kernel/m*
dtype0*
_output_shapes
: 

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
dtype0*
_output_shapes

:
~
Adam/dense_9/bias/mVarHandleOp*
shape:*$
shared_nameAdam/dense_9/bias/m*
dtype0*
_output_shapes
: 
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
dtype0*
_output_shapes
:
Ж
Adam/lstm_18/kernel/mVarHandleOp*
shape
:P*&
shared_nameAdam/lstm_18/kernel/m*
dtype0*
_output_shapes
: 

)Adam/lstm_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm_18/kernel/m*
dtype0*
_output_shapes

:P
Ъ
Adam/lstm_18/recurrent_kernel/mVarHandleOp*
shape
:P*0
shared_name!Adam/lstm_18/recurrent_kernel/m*
dtype0*
_output_shapes
: 
У
3Adam/lstm_18/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm_18/recurrent_kernel/m*
dtype0*
_output_shapes

:P
~
Adam/lstm_18/bias/mVarHandleOp*
shape:P*$
shared_nameAdam/lstm_18/bias/m*
dtype0*
_output_shapes
: 
w
'Adam/lstm_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_18/bias/m*
dtype0*
_output_shapes
:P
Ж
Adam/lstm_19/kernel/mVarHandleOp*
shape
:P*&
shared_nameAdam/lstm_19/kernel/m*
dtype0*
_output_shapes
: 

)Adam/lstm_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm_19/kernel/m*
dtype0*
_output_shapes

:P
Ъ
Adam/lstm_19/recurrent_kernel/mVarHandleOp*
shape
:P*0
shared_name!Adam/lstm_19/recurrent_kernel/m*
dtype0*
_output_shapes
: 
У
3Adam/lstm_19/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm_19/recurrent_kernel/m*
dtype0*
_output_shapes

:P
~
Adam/lstm_19/bias/mVarHandleOp*
shape:P*$
shared_nameAdam/lstm_19/bias/m*
dtype0*
_output_shapes
: 
w
'Adam/lstm_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_19/bias/m*
dtype0*
_output_shapes
:P
Ж
Adam/dense_9/kernel/vVarHandleOp*
shape
:*&
shared_nameAdam/dense_9/kernel/v*
dtype0*
_output_shapes
: 

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
dtype0*
_output_shapes

:
~
Adam/dense_9/bias/vVarHandleOp*
shape:*$
shared_nameAdam/dense_9/bias/v*
dtype0*
_output_shapes
: 
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
dtype0*
_output_shapes
:
Ж
Adam/lstm_18/kernel/vVarHandleOp*
shape
:P*&
shared_nameAdam/lstm_18/kernel/v*
dtype0*
_output_shapes
: 

)Adam/lstm_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm_18/kernel/v*
dtype0*
_output_shapes

:P
Ъ
Adam/lstm_18/recurrent_kernel/vVarHandleOp*
shape
:P*0
shared_name!Adam/lstm_18/recurrent_kernel/v*
dtype0*
_output_shapes
: 
У
3Adam/lstm_18/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm_18/recurrent_kernel/v*
dtype0*
_output_shapes

:P
~
Adam/lstm_18/bias/vVarHandleOp*
shape:P*$
shared_nameAdam/lstm_18/bias/v*
dtype0*
_output_shapes
: 
w
'Adam/lstm_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_18/bias/v*
dtype0*
_output_shapes
:P
Ж
Adam/lstm_19/kernel/vVarHandleOp*
shape
:P*&
shared_nameAdam/lstm_19/kernel/v*
dtype0*
_output_shapes
: 

)Adam/lstm_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm_19/kernel/v*
dtype0*
_output_shapes

:P
Ъ
Adam/lstm_19/recurrent_kernel/vVarHandleOp*
shape
:P*0
shared_name!Adam/lstm_19/recurrent_kernel/v*
dtype0*
_output_shapes
: 
У
3Adam/lstm_19/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm_19/recurrent_kernel/v*
dtype0*
_output_shapes

:P
~
Adam/lstm_19/bias/vVarHandleOp*
shape:P*$
shared_nameAdam/lstm_19/bias/v*
dtype0*
_output_shapes
: 
w
'Adam/lstm_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_19/bias/v*
dtype0*
_output_shapes
:P

NoOpNoOp
і)
ConstConst"/device:CPU:0*п(
valueе(Bв( Bџ(
у
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer

signatures
trainable_variables
	variables
	regularization_losses

	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
–
!iter

"beta_1

#beta_2
	$decay
%learning_ratemBmC&mD'mE(mF)mG*mH+mIvJvK&vL'vM(vN)vO*vP+vQ
 
8
&0
'1
(2
)3
*4
+5
6
7
8
&0
'1
(2
)3
*4
+5
6
7
 
Ъ
trainable_variables
,metrics
	variables
-layer_regularization_losses

.layers
/non_trainable_variables
	regularization_losses
 
 
 
Ъ
trainable_variables
0metrics
	variables
1layer_regularization_losses

2layers
3non_trainable_variables
regularization_losses
;

&kernel
'recurrent_kernel
(bias
4	keras_api
 

&0
'1
(2

&0
'1
(2
 
Ъ
trainable_variables
5metrics
	variables
6layer_regularization_losses

7layers
8non_trainable_variables
regularization_losses
;

)kernel
*recurrent_kernel
+bias
9	keras_api
 

)0
*1
+2

)0
*1
+2
 
Ъ
trainable_variables
:metrics
	variables
;layer_regularization_losses

<layers
=non_trainable_variables
regularization_losses
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Ъ
trainable_variables
>metrics
	variables
?layer_regularization_losses

@layers
Anon_trainable_variables
regularization_losses
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
TR
VARIABLE_VALUElstm_18/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUElstm_18/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUElstm_18/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUElstm_19/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUElstm_19/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUElstm_19/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
2
 
 
 
 
 
 
 
 

0
 
 
 
 

0
 
 
 
 
 
}{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/lstm_18/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/lstm_18/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/lstm_18/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/lstm_19/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/lstm_19/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/lstm_19/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/lstm_18/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/lstm_18/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/lstm_18/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/lstm_19/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/lstm_19/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/lstm_19/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
И
serving_default_lstm_18_inputPlaceholder* 
shape:€€€€€€€€€
*
dtype0*+
_output_shapes
:€€€€€€€€€

Ј
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_18_inputlstm_18/kernellstm_18/recurrent_kernellstm_18/biaslstm_19/kernellstm_19/recurrent_kernellstm_19/biasdense_9/kerneldense_9/bias*-
_gradient_op_typePartitionedCall-359698*-
f(R&
$__inference_signature_wrapper_359633*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2	*'
_output_shapes
:€€€€€€€€€
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
∆
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp"lstm_18/kernel/Read/ReadVariableOp,lstm_18/recurrent_kernel/Read/ReadVariableOp lstm_18/bias/Read/ReadVariableOp"lstm_19/kernel/Read/ReadVariableOp,lstm_19/recurrent_kernel/Read/ReadVariableOp lstm_19/bias/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp)Adam/lstm_18/kernel/m/Read/ReadVariableOp3Adam/lstm_18/recurrent_kernel/m/Read/ReadVariableOp'Adam/lstm_18/bias/m/Read/ReadVariableOp)Adam/lstm_19/kernel/m/Read/ReadVariableOp3Adam/lstm_19/recurrent_kernel/m/Read/ReadVariableOp'Adam/lstm_19/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp)Adam/lstm_18/kernel/v/Read/ReadVariableOp3Adam/lstm_18/recurrent_kernel/v/Read/ReadVariableOp'Adam/lstm_18/bias/v/Read/ReadVariableOp)Adam/lstm_19/kernel/v/Read/ReadVariableOp3Adam/lstm_19/recurrent_kernel/v/Read/ReadVariableOp'Adam/lstm_19/bias/v/Read/ReadVariableOpConst*-
_gradient_op_typePartitionedCall-359749*(
f#R!
__inference__traced_save_359748*
Tout
2**
config_proto

CPU

GPU 2J 8**
Tin#
!2	*
_output_shapes
: 
э
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_9/kerneldense_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_18/kernellstm_18/recurrent_kernellstm_18/biaslstm_19/kernellstm_19/recurrent_kernellstm_19/biasAdam/dense_9/kernel/mAdam/dense_9/bias/mAdam/lstm_18/kernel/mAdam/lstm_18/recurrent_kernel/mAdam/lstm_18/bias/mAdam/lstm_19/kernel/mAdam/lstm_19/recurrent_kernel/mAdam/lstm_19/bias/mAdam/dense_9/kernel/vAdam/dense_9/bias/vAdam/lstm_18/kernel/vAdam/lstm_18/recurrent_kernel/vAdam/lstm_18/bias/vAdam/lstm_19/kernel/vAdam/lstm_19/recurrent_kernel/vAdam/lstm_19/bias/v*-
_gradient_op_typePartitionedCall-359849*+
f&R$
"__inference__traced_restore_359848*
Tout
2**
config_proto

CPU

GPU 2J 8*)
Tin"
 2*
_output_shapes
: Вы
Щ
§
$__inference_while_cond_318423_346547
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : :  : : :	 : :
 
№+
В
$__inference_while_body_323435_343993
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€§
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:PН
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P®
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pt
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PҐ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
ЈG
Ц
 __inference_standard_lstm_349713

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
€€€€€€€€€B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
_lower_using_switch_merge(*
parallel_iterations *0
cond(R&
$__inference_while_cond_319864_349626*
_num_original_outputs*0
body(R&
$__inference_while_body_319865_343556*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€
[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ≥
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€∞

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:€€€€€€€€€
ґ

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€ґ

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*\
_input_shapesK
I:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
д
ќ
)__inference_restored_function_body_356908

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCall∆
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-346411*L
fGRE
C__inference_lstm_19_layer_call_and_return_conditional_losses_346410*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*а
_output_shapesЌ
 :€€€€€€€€€:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€: :
€€€€€€€€€::::: : : : : : : : : : : : : : : : : : : : : : :
€€€€€€€€€:В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
м
§
$__inference_while_cond_316034_352877
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : :  : : :	 : :
 
µ
Ы
__inference_loss_fn_0_344777:
6dense_9_kernel_regularizer_abs_readvariableop_resource
identityИҐ-dense_9/kernel/Regularizer/Abs/ReadVariableOp“
-dense_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_9_kernel_regularizer_abs_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Е
dense_9/kernel/Regularizer/AbsAbs5dense_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:q
 dense_9/kernel/Regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:Х
dense_9/kernel/Regularizer/SumSum"dense_9/kernel/Regularizer/Abs:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/mul/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: Ъ
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: Ч
dense_9/kernel/Regularizer/addAddV2)dense_9/kernel/Regularizer/add/x:output:0"dense_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: Й
IdentityIdentity"dense_9/kernel/Regularizer/add:z:0.^dense_9/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
:2^
-dense_9/kernel/Regularizer/Abs/ReadVariableOp-dense_9/kernel/Regularizer/Abs/ReadVariableOp:  
№+
В
$__inference_while_body_325355_346289
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€§
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:PН
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P®
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pt
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PҐ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
Є
ѕ
(__inference_lstm_19_layer_call_fn_354614
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-354606*L
fGRE
C__inference_lstm_19_layer_call_and_return_conditional_losses_354605*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall: :( $
"
_user_specified_name
inputs/0: : 
“
ѕ
(__inference_lstm_18_layer_call_fn_356060
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-356052*L
fGRE
C__inference_lstm_18_layer_call_and_return_conditional_losses_356051*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€П
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall: :( $
"
_user_specified_name
inputs/0: : 
нG
Ц
 __inference_standard_lstm_350350

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
_lower_using_switch_merge(*
parallel_iterations *0
cond(R&
$__inference_while_cond_324396_345906*
_num_original_outputs*0
body(R&
$__inference_while_body_324397_350263*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ≥
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€є

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ґ

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€ґ

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*e
_input_shapesT
R:€€€€€€€€€€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
Б3
щ	
A__forward_lstm_19_layer_call_and_return_conditional_losses_357814

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity
statefulpartitionedcall
statefulpartitionedcall_0
statefulpartitionedcall_1
statefulpartitionedcall_2
statefulpartitionedcall_3
statefulpartitionedcall_4
statefulpartitionedcall_5
statefulpartitionedcall_6
statefulpartitionedcall_7
statefulpartitionedcall_8
statefulpartitionedcall_9
statefulpartitionedcall_10
statefulpartitionedcall_11
statefulpartitionedcall_12
statefulpartitionedcall_13
statefulpartitionedcall_14
statefulpartitionedcall_15
statefulpartitionedcall_16
statefulpartitionedcall_17
statefulpartitionedcall_18
statefulpartitionedcall_19
statefulpartitionedcall_20
statefulpartitionedcall_21
statefulpartitionedcall_22
statefulpartitionedcall_23
statefulpartitionedcall_24
statefulpartitionedcall_25
statefulpartitionedcall_26
statefulpartitionedcall_27
statefulpartitionedcall_28
statefulpartitionedcall_29
statefulpartitionedcall_30
statefulpartitionedcall_31ИҐStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ж
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-346377*'
f"R 
__forward_standard_lstm_357743*.
Tout&
$2"**
config_proto

CPU

GPU 2J 8*
Tin

2*а
_output_shapesЌ
 :€€€€€€€€€:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€: :
€€€€€€€€€::::: : : : : : : : : : : : : : : : : : : : : : :
€€€€€€€€€:В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"?
statefulpartitionedcall_11!StatefulPartitionedCall:output:13"?
statefulpartitionedcall_12!StatefulPartitionedCall:output:14"
identityIdentity:output:0"=
statefulpartitionedcall_0 StatefulPartitionedCall:output:2"?
statefulpartitionedcall_13!StatefulPartitionedCall:output:15"?
statefulpartitionedcall_14!StatefulPartitionedCall:output:16"=
statefulpartitionedcall_1 StatefulPartitionedCall:output:3"?
statefulpartitionedcall_20!StatefulPartitionedCall:output:22"?
statefulpartitionedcall_15!StatefulPartitionedCall:output:17"=
statefulpartitionedcall_2 StatefulPartitionedCall:output:4"?
statefulpartitionedcall_21!StatefulPartitionedCall:output:23"=
statefulpartitionedcall_3 StatefulPartitionedCall:output:5"?
statefulpartitionedcall_16!StatefulPartitionedCall:output:18"?
statefulpartitionedcall_22!StatefulPartitionedCall:output:24"=
statefulpartitionedcall_4 StatefulPartitionedCall:output:6"?
statefulpartitionedcall_17!StatefulPartitionedCall:output:19"?
statefulpartitionedcall_23!StatefulPartitionedCall:output:25"=
statefulpartitionedcall_5 StatefulPartitionedCall:output:7"?
statefulpartitionedcall_18!StatefulPartitionedCall:output:20"?
statefulpartitionedcall_24!StatefulPartitionedCall:output:26"=
statefulpartitionedcall_6 StatefulPartitionedCall:output:8"?
statefulpartitionedcall_19!StatefulPartitionedCall:output:21"=
statefulpartitionedcall_7 StatefulPartitionedCall:output:9"?
statefulpartitionedcall_25!StatefulPartitionedCall:output:27"?
statefulpartitionedcall_30!StatefulPartitionedCall:output:32">
statefulpartitionedcall_8!StatefulPartitionedCall:output:10"?
statefulpartitionedcall_26!StatefulPartitionedCall:output:28"?
statefulpartitionedcall_31!StatefulPartitionedCall:output:33">
statefulpartitionedcall_9!StatefulPartitionedCall:output:11"?
statefulpartitionedcall_27!StatefulPartitionedCall:output:29"?
statefulpartitionedcall_28!StatefulPartitionedCall:output:30"?
statefulpartitionedcall_29!StatefulPartitionedCall:output:31";
statefulpartitionedcall StatefulPartitionedCall:output:1"?
statefulpartitionedcall_10!StatefulPartitionedCall:output:12*q
backward_function_nameWU__inference___backward_lstm_19_layer_call_and_return_conditional_losses_357247_357815*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
ћq
а
"__inference__traced_restore_359848
file_prefix#
assignvariableop_dense_9_kernel#
assignvariableop_1_dense_9_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate%
!assignvariableop_7_lstm_18_kernel/
+assignvariableop_8_lstm_18_recurrent_kernel#
assignvariableop_9_lstm_18_bias&
"assignvariableop_10_lstm_19_kernel0
,assignvariableop_11_lstm_19_recurrent_kernel$
 assignvariableop_12_lstm_19_bias-
)assignvariableop_13_adam_dense_9_kernel_m+
'assignvariableop_14_adam_dense_9_bias_m-
)assignvariableop_15_adam_lstm_18_kernel_m7
3assignvariableop_16_adam_lstm_18_recurrent_kernel_m+
'assignvariableop_17_adam_lstm_18_bias_m-
)assignvariableop_18_adam_lstm_19_kernel_m7
3assignvariableop_19_adam_lstm_19_recurrent_kernel_m+
'assignvariableop_20_adam_lstm_19_bias_m-
)assignvariableop_21_adam_dense_9_kernel_v+
'assignvariableop_22_adam_dense_9_bias_v-
)assignvariableop_23_adam_lstm_18_kernel_v7
3assignvariableop_24_adam_lstm_18_recurrent_kernel_v+
'assignvariableop_25_adam_lstm_18_bias_v-
)assignvariableop_26_adam_lstm_19_kernel_v7
3assignvariableop_27_adam_lstm_19_recurrent_kernel_v+
'assignvariableop_28_adam_lstm_19_bias_v
identity_30ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ґ	RestoreV2ҐRestoreV2_1в
RestoreV2/tensor_namesConst"/device:CPU:0*И
valueюBыB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:™
RestoreV2/shape_and_slicesConst"/device:CPU:0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:∞
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*+
dtypes!
2	*И
_output_shapesv
t:::::::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:{
AssignVariableOpAssignVariableOpassignvariableop_dense_9_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_9_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0	*
_output_shapes
:|
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0*
dtype0	*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:~
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:~
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:}
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:Е
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:Б
AssignVariableOp_7AssignVariableOp!assignvariableop_7_lstm_18_kernelIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:Л
AssignVariableOp_8AssignVariableOp+assignvariableop_8_lstm_18_recurrent_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_lstm_18_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:Д
AssignVariableOp_10AssignVariableOp"assignvariableop_10_lstm_19_kernelIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:О
AssignVariableOp_11AssignVariableOp,assignvariableop_11_lstm_19_recurrent_kernelIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:В
AssignVariableOp_12AssignVariableOp assignvariableop_12_lstm_19_biasIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:Л
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_dense_9_kernel_mIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:Й
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_dense_9_bias_mIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:Л
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_lstm_18_kernel_mIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:Х
AssignVariableOp_16AssignVariableOp3assignvariableop_16_adam_lstm_18_recurrent_kernel_mIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:Й
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_lstm_18_bias_mIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:Л
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_lstm_19_kernel_mIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:Х
AssignVariableOp_19AssignVariableOp3assignvariableop_19_adam_lstm_19_recurrent_kernel_mIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:Й
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_lstm_19_bias_mIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:Л
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_9_kernel_vIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:Й
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_9_bias_vIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:Л
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_lstm_18_kernel_vIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:Х
AssignVariableOp_24AssignVariableOp3assignvariableop_24_adam_lstm_18_recurrent_kernel_vIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:Й
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_lstm_18_bias_vIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:Л
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_lstm_19_kernel_vIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:Х
AssignVariableOp_27AssignVariableOp3assignvariableop_27_adam_lstm_19_recurrent_kernel_vIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:Й
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_lstm_19_bias_vIdentity_28:output:0*
dtype0*
_output_shapes
 М
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:µ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ќ
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: Џ
Identity_30IdentityIdentity_29:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_30Identity_30:output:0*Й
_input_shapesx
v: :::::::::::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2: : : : : : :
 : : : : : :	 : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : : : 
±
Ќ
(__inference_lstm_18_layer_call_fn_353898

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-353838*L
fGRE
C__inference_lstm_18_layer_call_and_return_conditional_losses_353837*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:€€€€€€€€€
Ж
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€
"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
ы
z
)__inference_restored_function_body_356979"
statefulpartitionedcall_args_0
identityИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallstatefulpartitionedcall_args_0*-
_gradient_op_typePartitionedCall-344778*%
f R
__inference_loss_fn_0_344777*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*$
_output_shapes
: : : :q
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall:  
ї
и
C__inference_lstm_18_layer_call_and_return_conditional_losses_356051

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИҐStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ў
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-356018*)
f$R"
 __inference_standard_lstm_356017*
Tout	
2**
config_proto

CPU

GPU 2J 8*
Tin

2*o
_output_shapes]
[:€€€€€€€€€:€€€€€€€€€€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: П
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
ДА
Б
__forward_standard_lstm_358431

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4&
"tensorarrayv2stack_tensorliststack
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2
transpose_1_perm	
while
while_0
while_1
while_2
while_maximum_iterations
while_3
while_4
while_5
while_6
while_7
while_8
while_9
while_10
while_11
while_12
while_13
while_14
while_15
while_16
while_17
while_18
while_19
	transpose
transpose_permИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:B
transpose_0	Transposeinputstranspose/perm:output:0*
T0D
ShapeShapetranspose_0:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:ѕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose_0:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:л
strided_slice_1StridedSlicetranspose_0:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: м	
while_20Whilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbiasmul_2_0/accumulator:handle:0"placeholder_0/accumulator:handle:0 Sigmoid_2_0/accumulator:handle:0Tanh_1_0/accumulator:handle:0mul_0/accumulator:handle:0mul_1_0/accumulator:handle:0 Sigmoid_1_0/accumulator:handle:0$placeholder_3_0/accumulator:handle:0Sigmoid_0/accumulator:handle:0Tanh_0/accumulator:handle:0MatMul_0/accumulator:handle:0MatMul_1_0/accumulator:handle:0,MatMul/ReadVariableOp_0/accumulator:handle:0:TensorArrayV2Read/TensorListGetItem_0/accumulator:handle:0.MatMul_1/ReadVariableOp_0/accumulator:handle:0$placeholder_2_0/accumulator:handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*$
T
2*k
output_shapesZ
X: : : : :€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : *
_lower_using_switch_merge(*
parallel_iterations *:
cond2R0
.__inference_while_cond_323434_350093_rewritten*
_num_original_outputs*:
body2R0
.__inference_while_body_323435_343993_rewrittenN
while/IdentityIdentitywhile_20:output:0*
T0*
_output_shapes
: P
while/Identity_1Identitywhile_20:output:1*
T0*
_output_shapes
: P
while/Identity_2Identitywhile_20:output:2*
T0*
_output_shapes
: P
while/Identity_3Identitywhile_20:output:3*
T0*
_output_shapes
: a
while/Identity_4Identitywhile_20:output:4*
T0*'
_output_shapes
:€€€€€€€€€a
while/Identity_5Identitywhile_20:output:5*
T0*'
_output_shapes
:€€€€€€€€€P
while/Identity_6Identitywhile_20:output:6*
T0*
_output_shapes
: P
while/Identity_7Identitywhile_20:output:7*
T0*
_output_shapes
: P
while/Identity_8Identitywhile_20:output:8*
T0*
_output_shapes
: P
while/Identity_9Identitywhile_20:output:9*
T0*
_output_shapes
: R
while/Identity_10Identitywhile_20:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€
[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ґ
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp	^while_20*
T0*'
_output_shapes
:€€€€€€€€€≥

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp	^while_20*
T0*+
_output_shapes
:€€€€€€€€€
є

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp	^while_20*
T0*'
_output_shapes
:€€€€€€€€€є

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp	^while_20*
T0*'
_output_shapes
:€€€€€€€€€Я

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp	^while_20*
T0*
_output_shapes
: r
!mul_2_0/accumulator/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:і
mul_2_0/accumulatorEmptyTensorList*mul_2_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: J
Const_1Const*
valueB *
dtype0*
_output_shapes
: †
placeholder_0/accumulatorEmptyTensorListConst_1:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: v
%Sigmoid_2_0/accumulator/element_shapeConst*
valueB"€€€€€€€€*
dtype0*
_output_shapes
:Љ
Sigmoid_2_0/accumulatorEmptyTensorList.Sigmoid_2_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: s
"Tanh_1_0/accumulator/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:ґ
Tanh_1_0/accumulatorEmptyTensorList+Tanh_1_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: p
mul_0/accumulator/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:∞
mul_0/accumulatorEmptyTensorList(mul_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: r
!mul_1_0/accumulator/element_shapeConst*
valueB"€€€€€€€€*
dtype0*
_output_shapes
:і
mul_1_0/accumulatorEmptyTensorList*mul_1_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: v
%Sigmoid_1_0/accumulator/element_shapeConst*
valueB"€€€€€€€€*
dtype0*
_output_shapes
:Љ
Sigmoid_1_0/accumulatorEmptyTensorList.Sigmoid_1_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: z
)placeholder_3_0/accumulator/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:ƒ
placeholder_3_0/accumulatorEmptyTensorList2placeholder_3_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: t
#Sigmoid_0/accumulator/element_shapeConst*
valueB"€€€€€€€€*
dtype0*
_output_shapes
:Є
Sigmoid_0/accumulatorEmptyTensorList,Sigmoid_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: q
 Tanh_0/accumulator/element_shapeConst*
valueB"€€€€€€€€*
dtype0*
_output_shapes
:≤
Tanh_0/accumulatorEmptyTensorList)Tanh_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: s
"MatMul_0/accumulator/element_shapeConst*
valueB"€€€€€€€€*
dtype0*
_output_shapes
:ґ
MatMul_0/accumulatorEmptyTensorList+MatMul_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: u
$MatMul_1_0/accumulator/element_shapeConst*
valueB"€€€€€€€€*
dtype0*
_output_shapes
:Ї
MatMul_1_0/accumulatorEmptyTensorList-MatMul_1_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: |
1MatMul/ReadVariableOp_0/accumulator/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: ‘
#MatMul/ReadVariableOp_0/accumulatorEmptyTensorList:MatMul/ReadVariableOp_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Р
?TensorArrayV2Read/TensorListGetItem_0/accumulator/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:р
1TensorArrayV2Read/TensorListGetItem_0/accumulatorEmptyTensorListHTensorArrayV2Read/TensorListGetItem_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ~
3MatMul_1/ReadVariableOp_0/accumulator/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Ў
%MatMul_1/ReadVariableOp_0/accumulatorEmptyTensorList<MatMul_1/ReadVariableOp_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: z
)placeholder_2_0/accumulator/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:ƒ
placeholder_2_0/accumulatorEmptyTensorList2placeholder_2_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: "!

identity_3Identity_3:output:0"Q
"tensorarrayv2stack_tensorliststack+TensorArrayV2Stack/TensorListStack:tensor:0"!

identity_4Identity_4:output:0"
whilewhile_20:output:7"
identityIdentity:output:0"=
while_maximum_iterations!while/maximum_iterations:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"
while_0while_20:output:8"
while_1while_20:output:9"
while_2while_20:output:10"
while_3while_20:output:0"
while_4while_20:output:11"
while_10while_20:output:17"
while_5while_20:output:12"
while_11while_20:output:18")
transpose_permtranspose/perm:output:0"
while_6while_20:output:13"
while_12while_20:output:19"
while_7while_20:output:14"
while_13while_20:output:20"
while_8while_20:output:15"
while_14while_20:output:21"
while_9while_20:output:16"
while_15while_20:output:22"
while_16while_20:output:23"
while_17while_20:output:24"
while_18while_20:output:25"
while_19while_20:output:26"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*N
backward_function_name42__inference___backward_standard_lstm_357947_358432*\
_input_shapesK
I:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile_2020
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
№+
В
$__inference_while_body_317461_344600
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€§
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:PН
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P®
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pt
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PҐ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
№+
В
$__inference_while_body_318424_347327
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€§
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:PН
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P®
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pt
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PҐ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
ґ
к
C__inference_lstm_19_layer_call_and_return_conditional_losses_350384
inputs_0"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИҐStatefulPartitionedCall=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€џ
StatefulPartitionedCallStatefulPartitionedCallinputs_0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-350351*)
f$R"
 __inference_standard_lstm_350350*
Tout	
2**
config_proto

CPU

GPU 2J 8*
Tin

2*o
_output_shapes]
[:€€€€€€€€€:€€€€€€€€€€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall: :( $
"
_user_specified_name
inputs/0: : 
“
ѕ
(__inference_lstm_18_layer_call_fn_353007
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-352999*L
fGRE
C__inference_lstm_18_layer_call_and_return_conditional_losses_352998*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€П
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall: :( $
"
_user_specified_name
inputs/0: : 
Ю
Ь
!__inference__wrapped_model_359518
lstm_18_input7
3sequential_9_lstm_18_statefulpartitionedcall_args_17
3sequential_9_lstm_18_statefulpartitionedcall_args_27
3sequential_9_lstm_18_statefulpartitionedcall_args_37
3sequential_9_lstm_19_statefulpartitionedcall_args_17
3sequential_9_lstm_19_statefulpartitionedcall_args_27
3sequential_9_lstm_19_statefulpartitionedcall_args_37
3sequential_9_dense_9_statefulpartitionedcall_args_17
3sequential_9_dense_9_statefulpartitionedcall_args_2
identityИҐ,sequential_9/dense_9/StatefulPartitionedCallҐ,sequential_9/lstm_18/StatefulPartitionedCallҐ,sequential_9/lstm_19/StatefulPartitionedCall—
,sequential_9/lstm_18/StatefulPartitionedCallStatefulPartitionedCalllstm_18_input3sequential_9_lstm_18_statefulpartitionedcall_args_13sequential_9_lstm_18_statefulpartitionedcall_args_23sequential_9_lstm_18_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-356884*2
f-R+
)__inference_restored_function_body_356883*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:€€€€€€€€€
х
,sequential_9/lstm_19/StatefulPartitionedCallStatefulPartitionedCall5sequential_9/lstm_18/StatefulPartitionedCall:output:03sequential_9_lstm_19_statefulpartitionedcall_args_13sequential_9_lstm_19_statefulpartitionedcall_args_23sequential_9_lstm_19_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-356925*2
f-R+
)__inference_restored_function_body_356924*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€њ
,sequential_9/dense_9/StatefulPartitionedCallStatefulPartitionedCall5sequential_9/lstm_19/StatefulPartitionedCall:output:03sequential_9_dense_9_statefulpartitionedcall_args_13sequential_9_dense_9_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-356946*2
f-R+
)__inference_restored_function_body_356945*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€К
IdentityIdentity5sequential_9/dense_9/StatefulPartitionedCall:output:0-^sequential_9/dense_9/StatefulPartitionedCall-^sequential_9/lstm_18/StatefulPartitionedCall-^sequential_9/lstm_19/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*J
_input_shapes9
7:€€€€€€€€€
::::::::2\
,sequential_9/lstm_18/StatefulPartitionedCall,sequential_9/lstm_18/StatefulPartitionedCall2\
,sequential_9/lstm_19/StatefulPartitionedCall,sequential_9/lstm_19/StatefulPartitionedCall2\
,sequential_9/dense_9/StatefulPartitionedCall,sequential_9/dense_9/StatefulPartitionedCall: : : : : :- )
'
_user_specified_namelstm_18_input: : : 
м
§
$__inference_while_cond_324396_345906
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : :  : : :	 : :
 
м
§
$__inference_while_cond_317460_354484
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : :  : : :	 : :
 
ї
и
C__inference_lstm_18_layer_call_and_return_conditional_losses_352998

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИҐStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ў
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-352965*)
f$R"
 __inference_standard_lstm_352964*
Tout	
2**
config_proto

CPU

GPU 2J 8*
Tin

2*o
_output_shapes]
[:€€€€€€€€€:€€€€€€€€€€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: П
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
©
Ќ
(__inference_lstm_19_layer_call_fn_349756

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-349748*L
fGRE
C__inference_lstm_19_layer_call_and_return_conditional_losses_349747*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
нG
Ц
 __inference_standard_lstm_356017

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
_lower_using_switch_merge(*
parallel_iterations *0
cond(R&
$__inference_while_cond_315552_355930*
_num_original_outputs*0
body(R&
$__inference_while_body_315553_355732*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ≥
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€є

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ґ

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€ґ

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*e
_input_shapesT
R:€€€€€€€€€€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
†
и
C__inference_lstm_18_layer_call_and_return_conditional_losses_354092

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИҐStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€–
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-354059*)
f$R"
 __inference_standard_lstm_354058*
Tout	
2**
config_proto

CPU

GPU 2J 8*
Tin

2*f
_output_shapesT
R:€€€€€€€€€:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€: Ж
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€
"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
м
§
$__inference_while_cond_315552_355930
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : :  : : :	 : :
 
ь
М
C__inference_dense_9_layer_call_and_return_conditional_losses_349317

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ-dense_9/kernel/Regularizer/Abs/ReadVariableOpҐ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€“
-dense_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Е
dense_9/kernel/Regularizer/AbsAbs5dense_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:q
 dense_9/kernel/Regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:Х
dense_9/kernel/Regularizer/SumSum"dense_9/kernel/Regularizer/Abs:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/mul/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: Ъ
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: Ч
dense_9/kernel/Regularizer/addAddV2)dense_9/kernel/Regularizer/add/x:output:0"dense_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_9/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€::2^
-dense_9/kernel/Regularizer/Abs/ReadVariableOp-dense_9/kernel/Regularizer/Abs/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
и
ќ
)__inference_restored_function_body_356867

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCall∆
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-350215*L
fGRE
C__inference_lstm_18_layer_call_and_return_conditional_losses_350214*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*а
_output_shapesЌ
 :€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: :
€€€€€€€€€::::: : : : : : : : : : : : : : : : : : : : : : :
€€€€€€€€€:Ж
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€
"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
Щ
§
$__inference_while_cond_319864_349626
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : :  : : :	 : :
 
„G
Ц
 __inference_standard_lstm_346376

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
€€€€€€€€€B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: —
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
_lower_using_switch_merge(*
parallel_iterations *0
cond(R&
$__inference_while_cond_325354_344844*
_num_original_outputs*0
body(R&
$__inference_while_body_325355_346289*l
_output_shapesZ
X: : : : :€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€
[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ≥
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€∞

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:€€€€€€€€€
ґ

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€ґ

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*\
_input_shapesK
I:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
Щ
§
$__inference_while_cond_325827_345890
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : :  : : :	 : :
 
і

Н
-__inference_sequential_9_layer_call_fn_359584
lstm_18_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИҐStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-359573*2
f-R+
)__inference_restored_function_body_359572*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2	*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*J
_input_shapes9
7:€€€€€€€€€
::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :- )
'
_user_specified_namelstm_18_input: : : 
Ь
и
C__inference_lstm_19_layer_call_and_return_conditional_losses_345323

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИҐStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€–
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-345290*)
f$R"
 __inference_standard_lstm_345289*
Tout	
2**
config_proto

CPU

GPU 2J 8*
Tin

2*f
_output_shapesT
R:€€€€€€€€€:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€: В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
№+
В
$__inference_while_body_324397_350263
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€§
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:PН
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P®
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pt
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PҐ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
√
к
C__inference_lstm_18_layer_call_and_return_conditional_losses_349608
inputs_0"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИҐStatefulPartitionedCall=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€џ
StatefulPartitionedCallStatefulPartitionedCallinputs_0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-349575*)
f$R"
 __inference_standard_lstm_349574*
Tout	
2**
config_proto

CPU

GPU 2J 8*
Tin

2*o
_output_shapes]
[:€€€€€€€€€:€€€€€€€€€€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: П
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall: :( $
"
_user_specified_name
inputs/0: : 
Х
†
.__inference_while_cond_323434_350093_rewritten
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
mul_2_0_accumulator
placeholder_0_accumulator
sigmoid_2_0_accumulator
tanh_1_0_accumulator
mul_0_accumulator
mul_1_0_accumulator
sigmoid_1_0_accumulator
placeholder_3_0_accumulator
sigmoid_0_accumulator
tanh_0_accumulator
matmul_0_accumulator
matmul_1_0_accumulator'
#matmul_readvariableop_0_accumulator5
1tensorarrayv2read_tensorlistgetitem_0_accumulator)
%matmul_1_readvariableop_0_accumulator
placeholder_2_0_accumulator
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*q
_input_shapes`
^: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : : : : : : : : : : : : : : : : : :
 : : : : : :	 : : : : :  : : : : : : : : : : 
„G
Ц
 __inference_standard_lstm_350180

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
€€€€€€€€€B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: —
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
_lower_using_switch_merge(*
parallel_iterations *0
cond(R&
$__inference_while_cond_323434_350093*
_num_original_outputs*0
body(R&
$__inference_while_body_323435_343993*l
_output_shapesZ
X: : : : :€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€
[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ≥
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€∞

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:€€€€€€€€€
ґ

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€ґ

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*\
_input_shapesK
I:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
№+
В
$__inference_while_body_323908_353971
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€§
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:PН
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P®
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pt
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PҐ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
Г
Ё
H__inference_sequential_9_layer_call_and_return_conditional_losses_347678

inputs*
&lstm_18_statefulpartitionedcall_args_1*
&lstm_18_statefulpartitionedcall_args_2*
&lstm_18_statefulpartitionedcall_args_3*
&lstm_19_statefulpartitionedcall_args_1*
&lstm_19_statefulpartitionedcall_args_2*
&lstm_19_statefulpartitionedcall_args_3*
&dense_9_statefulpartitionedcall_args_1*
&dense_9_statefulpartitionedcall_args_2
identityИҐdense_9/StatefulPartitionedCallҐ-dense_9/kernel/Regularizer/Abs/ReadVariableOpҐlstm_18/StatefulPartitionedCallҐlstm_19/StatefulPartitionedCall∞
lstm_18/StatefulPartitionedCallStatefulPartitionedCallinputs&lstm_18_statefulpartitionedcall_args_1&lstm_18_statefulpartitionedcall_args_2&lstm_18_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-347449*L
fGRE
C__inference_lstm_18_layer_call_and_return_conditional_losses_347448*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:€€€€€€€€€
ќ
lstm_19/StatefulPartitionedCallStatefulPartitionedCall(lstm_18/StatefulPartitionedCall:output:0&lstm_19_statefulpartitionedcall_args_1&lstm_19_statefulpartitionedcall_args_2&lstm_19_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-345324*L
fGRE
C__inference_lstm_19_layer_call_and_return_conditional_losses_345323*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€•
dense_9/StatefulPartitionedCallStatefulPartitionedCall(lstm_19/StatefulPartitionedCall:output:0&dense_9_statefulpartitionedcall_args_1&dense_9_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-347647*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_347646*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€д
-dense_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_9_statefulpartitionedcall_args_1 ^dense_9/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Е
dense_9/kernel/Regularizer/AbsAbs5dense_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:q
 dense_9/kernel/Regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:Х
dense_9/kernel/Regularizer/SumSum"dense_9/kernel/Regularizer/Abs:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/mul/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: Ъ
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: Ч
dense_9/kernel/Regularizer/addAddV2)dense_9/kernel/Regularizer/add/x:output:0"dense_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: Ж
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_9/StatefulPartitionedCall.^dense_9/kernel/Regularizer/Abs/ReadVariableOp ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*J
_input_shapes9
7:€€€€€€€€€
::::::::2^
-dense_9/kernel/Regularizer/Abs/ReadVariableOp-dense_9/kernel/Regularizer/Abs/ReadVariableOp2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
ЈG
Ц
 __inference_standard_lstm_345289

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
€€€€€€€€€B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
_lower_using_switch_merge(*
parallel_iterations *0
cond(R&
$__inference_while_cond_319391_344764*
_num_original_outputs*0
body(R&
$__inference_while_body_319392_345202*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€
[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ≥
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€∞

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:€€€€€€€€€
ґ

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€ґ

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*\
_input_shapesK
I:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
ЈG
Ц
 __inference_standard_lstm_350726

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
€€€€€€€€€B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
_lower_using_switch_merge(*
parallel_iterations *0
cond(R&
$__inference_while_cond_325827_345890*
_num_original_outputs*0
body(R&
$__inference_while_body_325828_350639*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€
[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ≥
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€∞

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:€€€€€€€€€
ґ

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€ґ

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*\
_input_shapesK
I:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
÷
©
(__inference_dense_9_layer_call_fn_347654

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-347647*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_347646*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
м
§
$__inference_while_cond_317942_344616
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : :  : : :	 : :
 
м
§
$__inference_while_cond_324867_346426
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : :  : : :	 : :
 
нG
Ц
 __inference_standard_lstm_346935

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
_lower_using_switch_merge(*
parallel_iterations *0
cond(R&
$__inference_while_cond_322947_346848*
_num_original_outputs*0
body(R&
$__inference_while_body_322948_346058*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ≥
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€є

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ґ

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€ґ

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*e
_input_shapesT
R:€€€€€€€€€€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
Ѓ
и
C__inference_lstm_19_layer_call_and_return_conditional_losses_354605

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИҐStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ў
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-354572*)
f$R"
 __inference_standard_lstm_354571*
Tout	
2**
config_proto

CPU

GPU 2J 8*
Tin

2*o
_output_shapes]
[:€€€€€€€€€:€€€€€€€€€€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
Х
†
.__inference_while_cond_325354_344844_rewritten
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
mul_2_0_accumulator
placeholder_0_accumulator
sigmoid_2_0_accumulator
tanh_1_0_accumulator
mul_0_accumulator
mul_1_0_accumulator
sigmoid_1_0_accumulator
placeholder_3_0_accumulator
sigmoid_0_accumulator
tanh_0_accumulator
matmul_0_accumulator
matmul_1_0_accumulator'
#matmul_readvariableop_0_accumulator5
1tensorarrayv2read_tensorlistgetitem_0_accumulator)
%matmul_1_readvariableop_0_accumulator
placeholder_2_0_accumulator
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*q
_input_shapes`
^: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : : : : : : : : : : : : : : : : : :
 : : : : : :	 : : : : :  : : : : : : : : : : 
Ь
и
C__inference_lstm_19_layer_call_and_return_conditional_losses_350760

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИҐStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€–
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-350727*)
f$R"
 __inference_standard_lstm_350726*
Tout	
2**
config_proto

CPU

GPU 2J 8*
Tin

2*f
_output_shapesT
R:€€€€€€€€€:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€: В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
Њ

Ж
-__inference_sequential_9_layer_call_fn_347706

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИҐStatefulPartitionedCallґ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-347679*Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_347678*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2	*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*J
_input_shapes9
7:€€€€€€€€€
::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
№+
В
$__inference_while_body_322948_346058
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€§
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:PН
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P®
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pt
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PҐ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
№+
В
$__inference_while_body_324868_354347
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€§
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:PН
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P®
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pt
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PҐ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
і

Н
-__inference_sequential_9_layer_call_fn_359613
lstm_18_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИҐStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-359602*2
f-R+
)__inference_restored_function_body_359601*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2	*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*J
_input_shapes9
7:€€€€€€€€€
::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :- )
'
_user_specified_namelstm_18_input: : : 
ЈG
Ц
 __inference_standard_lstm_354058

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
€€€€€€€€€B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
_lower_using_switch_merge(*
parallel_iterations *0
cond(R&
$__inference_while_cond_323907_344399*
_num_original_outputs*0
body(R&
$__inference_while_body_323908_353971*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€
[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ≥
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€∞

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:€€€€€€€€€
ґ

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€ґ

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*\
_input_shapesK
I:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
Я

В
)__inference_restored_function_body_359572

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИҐStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-347707*6
f1R/
-__inference_sequential_9_layer_call_fn_347706*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2	*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*J
_input_shapes9
7:€€€€€€€€€
::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
№+
В
$__inference_while_body_319865_343556
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€§
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:PН
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P®
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pt
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PҐ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
№+
В
$__inference_while_body_317943_344448
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€§
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:PН
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P®
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pt
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PҐ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
№+
В
$__inference_while_body_319392_345202
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€§
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:PН
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P®
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pt
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PҐ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
†
и
C__inference_lstm_18_layer_call_and_return_conditional_losses_353837

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИҐStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€–
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-353804*)
f$R"
 __inference_standard_lstm_353803*
Tout	
2**
config_proto

CPU

GPU 2J 8*
Tin

2*f
_output_shapesT
R:€€€€€€€€€:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€: Ж
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€
"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
ЈG
Ц
 __inference_standard_lstm_347414

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
€€€€€€€€€B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
_lower_using_switch_merge(*
parallel_iterations *0
cond(R&
$__inference_while_cond_318423_346547*
_num_original_outputs*0
body(R&
$__inference_while_body_318424_347327*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€
[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ≥
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€∞

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:€€€€€€€€€
ґ

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€ґ

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*\
_input_shapesK
I:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
№+
В
$__inference_while_body_316035_352861
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€§
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:PН
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P®
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pt
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PҐ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
№+
В
$__inference_while_body_318897_344826
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€§
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:PН
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P®
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pt
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PҐ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
√
к
C__inference_lstm_18_layer_call_and_return_conditional_losses_346969
inputs_0"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИҐStatefulPartitionedCall=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€џ
StatefulPartitionedCallStatefulPartitionedCallinputs_0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-346936*)
f$R"
 __inference_standard_lstm_346935*
Tout	
2**
config_proto

CPU

GPU 2J 8*
Tin

2*o
_output_shapes]
[:€€€€€€€€€:€€€€€€€€€€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: П
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall: :( $
"
_user_specified_name
inputs/0: : 
№+
В
$__inference_while_body_315553_355732
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€§
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:PН
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P®
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pt
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PҐ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
Щ
§
$__inference_while_cond_323434_350093
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : :  : : :	 : :
 
Я

В
)__inference_restored_function_body_359601

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИҐStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-353890*6
f1R/
-__inference_sequential_9_layer_call_fn_353889*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2	*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*J
_input_shapes9
7:€€€€€€€€€
::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
±
Ќ
(__inference_lstm_18_layer_call_fn_347457

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-347449*L
fGRE
C__inference_lstm_18_layer_call_and_return_conditional_losses_347448*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:€€€€€€€€€
Ж
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€
"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
†
и
C__inference_lstm_18_layer_call_and_return_conditional_losses_347448

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИҐStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€–
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-347415*)
f$R"
 __inference_standard_lstm_347414*
Tout	
2**
config_proto

CPU

GPU 2J 8*
Tin

2*f
_output_shapesT
R:€€€€€€€€€:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€: Ж
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€
"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
Є
ѕ
(__inference_lstm_19_layer_call_fn_344746
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-344738*L
fGRE
C__inference_lstm_19_layer_call_and_return_conditional_losses_344737*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall: :( $
"
_user_specified_name
inputs/0: : 
©
Ќ
(__inference_lstm_19_layer_call_fn_345332

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-345324*L
fGRE
C__inference_lstm_19_layer_call_and_return_conditional_losses_345323*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
£

Д
$__inference_signature_wrapper_359633
lstm_18_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-359622**
f%R#
!__inference__wrapped_model_359518*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2	*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*J
_input_shapes9
7:€€€€€€€€€
::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :- )
'
_user_specified_namelstm_18_input: : : 
Ј=
д
__inference__traced_save_359748
file_prefix-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop-
)savev2_lstm_18_kernel_read_readvariableop7
3savev2_lstm_18_recurrent_kernel_read_readvariableop+
'savev2_lstm_18_bias_read_readvariableop-
)savev2_lstm_19_kernel_read_readvariableop7
3savev2_lstm_19_recurrent_kernel_read_readvariableop+
'savev2_lstm_19_bias_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop4
0savev2_adam_lstm_18_kernel_m_read_readvariableop>
:savev2_adam_lstm_18_recurrent_kernel_m_read_readvariableop2
.savev2_adam_lstm_18_bias_m_read_readvariableop4
0savev2_adam_lstm_19_kernel_m_read_readvariableop>
:savev2_adam_lstm_19_recurrent_kernel_m_read_readvariableop2
.savev2_adam_lstm_19_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop4
0savev2_adam_lstm_18_kernel_v_read_readvariableop>
:savev2_adam_lstm_18_recurrent_kernel_v_read_readvariableop2
.savev2_adam_lstm_18_bias_v_read_readvariableop4
0savev2_adam_lstm_19_kernel_v_read_readvariableop>
:savev2_adam_lstm_19_recurrent_kernel_v_read_readvariableop2
.savev2_adam_lstm_19_bias_v_read_readvariableop
savev2_1_const

identity_1ИҐMergeV2CheckpointsҐSaveV2ҐSaveV2_1О
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_382fc752f8c345aa8c57cdf2303e9361/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: я
SaveV2/tensor_namesConst"/device:CPU:0*И
valueюBыB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:І
SaveV2/shape_and_slicesConst"/device:CPU:0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:ђ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop)savev2_lstm_18_kernel_read_readvariableop3savev2_lstm_18_recurrent_kernel_read_readvariableop'savev2_lstm_18_bias_read_readvariableop)savev2_lstm_19_kernel_read_readvariableop3savev2_lstm_19_recurrent_kernel_read_readvariableop'savev2_lstm_19_bias_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop0savev2_adam_lstm_18_kernel_m_read_readvariableop:savev2_adam_lstm_18_recurrent_kernel_m_read_readvariableop.savev2_adam_lstm_18_bias_m_read_readvariableop0savev2_adam_lstm_19_kernel_m_read_readvariableop:savev2_adam_lstm_19_recurrent_kernel_m_read_readvariableop.savev2_adam_lstm_19_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop0savev2_adam_lstm_18_kernel_v_read_readvariableop:savev2_adam_lstm_18_recurrent_kernel_v_read_readvariableop.savev2_adam_lstm_18_bias_v_read_readvariableop0savev2_adam_lstm_19_kernel_v_read_readvariableop:savev2_adam_lstm_19_recurrent_kernel_v_read_readvariableop.savev2_adam_lstm_19_bias_v_read_readvariableop"/device:CPU:0*+
dtypes!
2	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: Ч
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Й
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:√
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 є
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:Ц
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*п
_input_shapesЁ
Џ: ::: : : : : :P:P:P:P:P:P:::P:P:P:P:P:P:::P:P:P:P:P:P: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: : : : : : :
 : : : : : : :	 : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : : : 
Е3
щ	
A__forward_lstm_18_layer_call_and_return_conditional_losses_358502

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity
statefulpartitionedcall
statefulpartitionedcall_0
statefulpartitionedcall_1
statefulpartitionedcall_2
statefulpartitionedcall_3
statefulpartitionedcall_4
statefulpartitionedcall_5
statefulpartitionedcall_6
statefulpartitionedcall_7
statefulpartitionedcall_8
statefulpartitionedcall_9
statefulpartitionedcall_10
statefulpartitionedcall_11
statefulpartitionedcall_12
statefulpartitionedcall_13
statefulpartitionedcall_14
statefulpartitionedcall_15
statefulpartitionedcall_16
statefulpartitionedcall_17
statefulpartitionedcall_18
statefulpartitionedcall_19
statefulpartitionedcall_20
statefulpartitionedcall_21
statefulpartitionedcall_22
statefulpartitionedcall_23
statefulpartitionedcall_24
statefulpartitionedcall_25
statefulpartitionedcall_26
statefulpartitionedcall_27
statefulpartitionedcall_28
statefulpartitionedcall_29
statefulpartitionedcall_30
statefulpartitionedcall_31ИҐStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ж
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-350181*'
f"R 
__forward_standard_lstm_358431*.
Tout&
$2"**
config_proto

CPU

GPU 2J 8*
Tin

2*а
_output_shapesЌ
 :€€€€€€€€€:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€: :
€€€€€€€€€::::: : : : : : : : : : : : : : : : : : : : : : :
€€€€€€€€€:Ж
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€
"?
statefulpartitionedcall_11!StatefulPartitionedCall:output:13"?
statefulpartitionedcall_12!StatefulPartitionedCall:output:14"=
statefulpartitionedcall_0 StatefulPartitionedCall:output:2"?
statefulpartitionedcall_13!StatefulPartitionedCall:output:15"
identityIdentity:output:0"?
statefulpartitionedcall_14!StatefulPartitionedCall:output:16"=
statefulpartitionedcall_1 StatefulPartitionedCall:output:3"?
statefulpartitionedcall_20!StatefulPartitionedCall:output:22"?
statefulpartitionedcall_15!StatefulPartitionedCall:output:17"=
statefulpartitionedcall_2 StatefulPartitionedCall:output:4"?
statefulpartitionedcall_21!StatefulPartitionedCall:output:23"=
statefulpartitionedcall_3 StatefulPartitionedCall:output:5"?
statefulpartitionedcall_16!StatefulPartitionedCall:output:18"?
statefulpartitionedcall_22!StatefulPartitionedCall:output:24"=
statefulpartitionedcall_4 StatefulPartitionedCall:output:6"?
statefulpartitionedcall_17!StatefulPartitionedCall:output:19"?
statefulpartitionedcall_23!StatefulPartitionedCall:output:25"=
statefulpartitionedcall_5 StatefulPartitionedCall:output:7"?
statefulpartitionedcall_18!StatefulPartitionedCall:output:20"?
statefulpartitionedcall_24!StatefulPartitionedCall:output:26"=
statefulpartitionedcall_6 StatefulPartitionedCall:output:8"?
statefulpartitionedcall_19!StatefulPartitionedCall:output:21"=
statefulpartitionedcall_7 StatefulPartitionedCall:output:9"?
statefulpartitionedcall_25!StatefulPartitionedCall:output:27"?
statefulpartitionedcall_30!StatefulPartitionedCall:output:32">
statefulpartitionedcall_8!StatefulPartitionedCall:output:10"?
statefulpartitionedcall_26!StatefulPartitionedCall:output:28"?
statefulpartitionedcall_31!StatefulPartitionedCall:output:33">
statefulpartitionedcall_9!StatefulPartitionedCall:output:11"?
statefulpartitionedcall_27!StatefulPartitionedCall:output:29"?
statefulpartitionedcall_28!StatefulPartitionedCall:output:30"?
statefulpartitionedcall_29!StatefulPartitionedCall:output:31";
statefulpartitionedcall StatefulPartitionedCall:output:0"?
statefulpartitionedcall_10!StatefulPartitionedCall:output:12*q
backward_function_nameWU__inference___backward_lstm_18_layer_call_and_return_conditional_losses_357935_358503*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
ДА
Б
__forward_standard_lstm_357743

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4&
"tensorarrayv2stack_tensorliststack
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2
transpose_1_perm	
while
while_0
while_1
while_2
while_maximum_iterations
while_3
while_4
while_5
while_6
while_7
while_8
while_9
while_10
while_11
while_12
while_13
while_14
while_15
while_16
while_17
while_18
while_19
	transpose
transpose_permИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:B
transpose_0	Transposeinputstranspose/perm:output:0*
T0D
ShapeShapetranspose_0:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:ѕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose_0:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:л
strided_slice_1StridedSlicetranspose_0:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: м	
while_20Whilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbiasmul_2_0/accumulator:handle:0"placeholder_0/accumulator:handle:0 Sigmoid_2_0/accumulator:handle:0Tanh_1_0/accumulator:handle:0mul_0/accumulator:handle:0mul_1_0/accumulator:handle:0 Sigmoid_1_0/accumulator:handle:0$placeholder_3_0/accumulator:handle:0Sigmoid_0/accumulator:handle:0Tanh_0/accumulator:handle:0MatMul_0/accumulator:handle:0MatMul_1_0/accumulator:handle:0,MatMul/ReadVariableOp_0/accumulator:handle:0:TensorArrayV2Read/TensorListGetItem_0/accumulator:handle:0.MatMul_1/ReadVariableOp_0/accumulator:handle:0$placeholder_2_0/accumulator:handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*$
T
2*k
output_shapesZ
X: : : : :€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : *
_lower_using_switch_merge(*
parallel_iterations *:
cond2R0
.__inference_while_cond_325354_344844_rewritten*
_num_original_outputs*:
body2R0
.__inference_while_body_325355_346289_rewrittenN
while/IdentityIdentitywhile_20:output:0*
T0*
_output_shapes
: P
while/Identity_1Identitywhile_20:output:1*
T0*
_output_shapes
: P
while/Identity_2Identitywhile_20:output:2*
T0*
_output_shapes
: P
while/Identity_3Identitywhile_20:output:3*
T0*
_output_shapes
: a
while/Identity_4Identitywhile_20:output:4*
T0*'
_output_shapes
:€€€€€€€€€a
while/Identity_5Identitywhile_20:output:5*
T0*'
_output_shapes
:€€€€€€€€€P
while/Identity_6Identitywhile_20:output:6*
T0*
_output_shapes
: P
while/Identity_7Identitywhile_20:output:7*
T0*
_output_shapes
: P
while/Identity_8Identitywhile_20:output:8*
T0*
_output_shapes
: P
while/Identity_9Identitywhile_20:output:9*
T0*
_output_shapes
: R
while/Identity_10Identitywhile_20:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€
[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ґ
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp	^while_20*
T0*'
_output_shapes
:€€€€€€€€€≥

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp	^while_20*
T0*+
_output_shapes
:€€€€€€€€€
є

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp	^while_20*
T0*'
_output_shapes
:€€€€€€€€€є

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp	^while_20*
T0*'
_output_shapes
:€€€€€€€€€Я

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp	^while_20*
T0*
_output_shapes
: r
!mul_2_0/accumulator/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:і
mul_2_0/accumulatorEmptyTensorList*mul_2_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: J
Const_1Const*
valueB *
dtype0*
_output_shapes
: †
placeholder_0/accumulatorEmptyTensorListConst_1:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: v
%Sigmoid_2_0/accumulator/element_shapeConst*
valueB"€€€€€€€€*
dtype0*
_output_shapes
:Љ
Sigmoid_2_0/accumulatorEmptyTensorList.Sigmoid_2_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: s
"Tanh_1_0/accumulator/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:ґ
Tanh_1_0/accumulatorEmptyTensorList+Tanh_1_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: p
mul_0/accumulator/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:∞
mul_0/accumulatorEmptyTensorList(mul_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: r
!mul_1_0/accumulator/element_shapeConst*
valueB"€€€€€€€€*
dtype0*
_output_shapes
:і
mul_1_0/accumulatorEmptyTensorList*mul_1_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: v
%Sigmoid_1_0/accumulator/element_shapeConst*
valueB"€€€€€€€€*
dtype0*
_output_shapes
:Љ
Sigmoid_1_0/accumulatorEmptyTensorList.Sigmoid_1_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: z
)placeholder_3_0/accumulator/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:ƒ
placeholder_3_0/accumulatorEmptyTensorList2placeholder_3_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: t
#Sigmoid_0/accumulator/element_shapeConst*
valueB"€€€€€€€€*
dtype0*
_output_shapes
:Є
Sigmoid_0/accumulatorEmptyTensorList,Sigmoid_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: q
 Tanh_0/accumulator/element_shapeConst*
valueB"€€€€€€€€*
dtype0*
_output_shapes
:≤
Tanh_0/accumulatorEmptyTensorList)Tanh_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: s
"MatMul_0/accumulator/element_shapeConst*
valueB"€€€€€€€€*
dtype0*
_output_shapes
:ґ
MatMul_0/accumulatorEmptyTensorList+MatMul_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: u
$MatMul_1_0/accumulator/element_shapeConst*
valueB"€€€€€€€€*
dtype0*
_output_shapes
:Ї
MatMul_1_0/accumulatorEmptyTensorList-MatMul_1_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: |
1MatMul/ReadVariableOp_0/accumulator/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: ‘
#MatMul/ReadVariableOp_0/accumulatorEmptyTensorList:MatMul/ReadVariableOp_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Р
?TensorArrayV2Read/TensorListGetItem_0/accumulator/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:р
1TensorArrayV2Read/TensorListGetItem_0/accumulatorEmptyTensorListHTensorArrayV2Read/TensorListGetItem_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ~
3MatMul_1/ReadVariableOp_0/accumulator/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Ў
%MatMul_1/ReadVariableOp_0/accumulatorEmptyTensorList<MatMul_1/ReadVariableOp_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: z
)placeholder_2_0/accumulator/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:ƒ
placeholder_2_0/accumulatorEmptyTensorList2placeholder_2_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"
while_0while_20:output:8"
while_1while_20:output:9"
while_2while_20:output:10"
while_3while_20:output:0"
while_4while_20:output:11"
while_10while_20:output:17"
while_5while_20:output:12"
while_11while_20:output:18")
transpose_permtranspose/perm:output:0"
while_6while_20:output:13"
while_12while_20:output:19"
while_7while_20:output:14"
while_13while_20:output:20"
while_8while_20:output:15"
while_14while_20:output:21"
while_9while_20:output:16"
while_15while_20:output:22"
while_16while_20:output:23"
while_17while_20:output:24"
while_18while_20:output:25"
while_19while_20:output:26"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"Q
"tensorarrayv2stack_tensorliststack+TensorArrayV2Stack/TensorListStack:tensor:0"!

identity_4Identity_4:output:0"
whilewhile_20:output:7"
identityIdentity:output:0"=
while_maximum_iterations!while/maximum_iterations:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0*N
backward_function_name42__inference___backward_standard_lstm_357259_357744*\
_input_shapesK
I:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile_2020
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
нG
Ц
 __inference_standard_lstm_354434

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
_lower_using_switch_merge(*
parallel_iterations *0
cond(R&
$__inference_while_cond_324867_346426*
_num_original_outputs*0
body(R&
$__inference_while_body_324868_354347*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ≥
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€є

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ґ

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€ґ

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*e
_input_shapesT
R:€€€€€€€€€€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
Щ
§
$__inference_while_cond_318896_353716
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : :  : : :	 : :
 
Ч
и
C__inference_lstm_19_layer_call_and_return_conditional_losses_346410

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИҐStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ћ
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-346377*)
f$R"
 __inference_standard_lstm_346376*
Tout	
2**
config_proto

CPU

GPU 2J 8*
Tin

2*а
_output_shapesЌ
 :€€€€€€€€€:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€: :
€€€€€€€€€::::: : : : : : : : : : : : : : : : : : : : : : :
€€€€€€€€€:В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
ю
Ц
__forward_loss_fn_0_357116:
6dense_9_kernel_regularizer_abs_readvariableop_resource
identity"
dense_9_kernel_regularizer_sum$
 dense_9_kernel_regularizer_mul_x1
-dense_9_kernel_regularizer_abs_readvariableopИҐ-dense_9/kernel/Regularizer/Abs/ReadVariableOp“
-dense_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_9_kernel_regularizer_abs_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Е
dense_9/kernel/Regularizer/AbsAbs5dense_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:q
 dense_9/kernel/Regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:Х
dense_9/kernel/Regularizer/SumSum"dense_9/kernel/Regularizer/Abs:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/mul/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: Ъ
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: Ч
dense_9/kernel/Regularizer/addAddV2)dense_9/kernel/Regularizer/add/x:output:0"dense_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: Й
IdentityIdentity"dense_9/kernel/Regularizer/add:z:0.^dense_9/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: "
identityIdentity:output:0"M
 dense_9_kernel_regularizer_mul_x)dense_9/kernel/Regularizer/mul/x:output:0"f
-dense_9_kernel_regularizer_abs_readvariableop5dense_9/kernel/Regularizer/Abs/ReadVariableOp:value:0"I
dense_9_kernel_regularizer_sum'dense_9/kernel/Regularizer/Sum:output:0*J
backward_function_name0.__inference___backward_loss_fn_0_357100_357117*
_input_shapes
:2^
-dense_9/kernel/Regularizer/Abs/ReadVariableOp-dense_9/kernel/Regularizer/Abs/ReadVariableOp:  
Ь
и
C__inference_lstm_19_layer_call_and_return_conditional_losses_349747

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИҐStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€–
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-349714*)
f$R"
 __inference_standard_lstm_349713*
Tout	
2**
config_proto

CPU

GPU 2J 8*
Tin

2*f
_output_shapesT
R:€€€€€€€€€:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€: В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
Г
Ё
H__inference_sequential_9_layer_call_and_return_conditional_losses_353861

inputs*
&lstm_18_statefulpartitionedcall_args_1*
&lstm_18_statefulpartitionedcall_args_2*
&lstm_18_statefulpartitionedcall_args_3*
&lstm_19_statefulpartitionedcall_args_1*
&lstm_19_statefulpartitionedcall_args_2*
&lstm_19_statefulpartitionedcall_args_3*
&dense_9_statefulpartitionedcall_args_1*
&dense_9_statefulpartitionedcall_args_2
identityИҐdense_9/StatefulPartitionedCallҐ-dense_9/kernel/Regularizer/Abs/ReadVariableOpҐlstm_18/StatefulPartitionedCallҐlstm_19/StatefulPartitionedCall∞
lstm_18/StatefulPartitionedCallStatefulPartitionedCallinputs&lstm_18_statefulpartitionedcall_args_1&lstm_18_statefulpartitionedcall_args_2&lstm_18_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-353838*L
fGRE
C__inference_lstm_18_layer_call_and_return_conditional_losses_353837*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:€€€€€€€€€
ќ
lstm_19/StatefulPartitionedCallStatefulPartitionedCall(lstm_18/StatefulPartitionedCall:output:0&lstm_19_statefulpartitionedcall_args_1&lstm_19_statefulpartitionedcall_args_2&lstm_19_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-349748*L
fGRE
C__inference_lstm_19_layer_call_and_return_conditional_losses_349747*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€•
dense_9/StatefulPartitionedCallStatefulPartitionedCall(lstm_19/StatefulPartitionedCall:output:0&dense_9_statefulpartitionedcall_args_1&dense_9_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-347647*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_347646*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€д
-dense_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_9_statefulpartitionedcall_args_1 ^dense_9/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Е
dense_9/kernel/Regularizer/AbsAbs5dense_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:q
 dense_9/kernel/Regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:Х
dense_9/kernel/Regularizer/SumSum"dense_9/kernel/Regularizer/Abs:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/mul/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: Ъ
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: Ч
dense_9/kernel/Regularizer/addAddV2)dense_9/kernel/Regularizer/add/x:output:0"dense_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: Ж
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_9/StatefulPartitionedCall.^dense_9/kernel/Regularizer/Abs/ReadVariableOp ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*J
_input_shapes9
7:€€€€€€€€€
::::::::2^
-dense_9/kernel/Regularizer/Abs/ReadVariableOp-dense_9/kernel/Regularizer/Abs/ReadVariableOp2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
≤
ќ
)__inference_restored_function_body_356883

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-354093*L
fGRE
C__inference_lstm_18_layer_call_and_return_conditional_losses_354092*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:€€€€€€€€€
Ж
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€
"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
ЧS
К
.__inference_while_body_325355_346289_rewritten
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0_0S
Otensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0_0&
"matmul_readvariableop_resource_0_0(
$matmul_1_readvariableop_resource_0_0'
#biasadd_readvariableop_resource_0_0*
&tensorlistpushback_mul_2_0_accumulator2
.tensorlistpushback_1_placeholder_0_accumulator0
,tensorlistpushback_2_sigmoid_2_0_accumulator-
)tensorlistpushback_3_tanh_1_0_accumulator*
&tensorlistpushback_4_mul_0_accumulator,
(tensorlistpushback_5_mul_1_0_accumulator0
,tensorlistpushback_6_sigmoid_1_0_accumulator4
0tensorlistpushback_7_placeholder_3_0_accumulator.
*tensorlistpushback_8_sigmoid_0_accumulator+
'tensorlistpushback_9_tanh_0_accumulator.
*tensorlistpushback_10_matmul_0_accumulator0
,tensorlistpushback_11_matmul_1_0_accumulator=
9tensorlistpushback_12_matmul_readvariableop_0_accumulatorK
Gtensorlistpushback_13_tensorarrayv2read_tensorlistgetitem_0_accumulator?
;tensorlistpushback_14_matmul_1_readvariableop_0_accumulator5
1tensorlistpushback_15_placeholder_2_0_accumulator
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
tensorlistpushback
tensorlistpushback_1
tensorlistpushback_2
tensorlistpushback_3
tensorlistpushback_4
tensorlistpushback_5
tensorlistpushback_6
tensorlistpushback_7
tensorlistpushback_8
tensorlistpushback_9
tensorlistpushback_10
tensorlistpushback_11
tensorlistpushback_12
tensorlistpushback_13
tensorlistpushback_14
tensorlistpushback_15ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Р
#TensorArrayV2Read/TensorListGetItemTensorListGetItemOtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€†
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_resource_0_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Ц
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€§
MatMul_1/ReadVariableOpReadVariableOp$matmul_1_readvariableop_resource_0_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:}
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€m
addAddV2MatMul:product:0MatMul_1:product:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Ґ
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_resource_0_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: џ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*Д
_output_shapesr
p:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€]
SigmoidSigmoidsplit:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€_
	Sigmoid_1Sigmoidsplit:output:1*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€W
TanhTanhsplit:output:2*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€_
	Sigmoid_2Sigmoidsplit:output:3*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€И
TensorListPushBackTensorListPushBack&tensorlistpushback_mul_2_0_accumulator	mul_2:z:0*
element_dtype0*
_output_shapes
: Ф
TensorListPushBack_1TensorListPushBack.tensorlistpushback_1_placeholder_0_accumulatorplaceholder*
element_dtype0*
_output_shapes
: Ф
TensorListPushBack_2TensorListPushBack,tensorlistpushback_2_sigmoid_2_0_accumulatorSigmoid_2:y:0*
element_dtype0*
_output_shapes
: О
TensorListPushBack_3TensorListPushBack)tensorlistpushback_3_tanh_1_0_accumulator
Tanh_1:y:0*
element_dtype0*
_output_shapes
: И
TensorListPushBack_4TensorListPushBack&tensorlistpushback_4_mul_0_accumulatormul:z:0*
element_dtype0*
_output_shapes
: М
TensorListPushBack_5TensorListPushBack(tensorlistpushback_5_mul_1_0_accumulator	mul_1:z:0*
element_dtype0*
_output_shapes
: Ф
TensorListPushBack_6TensorListPushBack,tensorlistpushback_6_sigmoid_1_0_accumulatorSigmoid_1:y:0*
element_dtype0*
_output_shapes
: Ш
TensorListPushBack_7TensorListPushBack0tensorlistpushback_7_placeholder_3_0_accumulatorplaceholder_3*
element_dtype0*
_output_shapes
: Р
TensorListPushBack_8TensorListPushBack*tensorlistpushback_8_sigmoid_0_accumulatorSigmoid:y:0*
element_dtype0*
_output_shapes
: К
TensorListPushBack_9TensorListPushBack'tensorlistpushback_9_tanh_0_accumulatorTanh:y:0*
element_dtype0*
_output_shapes
: Ц
TensorListPushBack_10TensorListPushBack*tensorlistpushback_10_matmul_0_accumulatorMatMul:product:0*
element_dtype0*
_output_shapes
: Ъ
TensorListPushBack_11TensorListPushBack,tensorlistpushback_11_matmul_1_0_accumulatorMatMul_1:product:0*
element_dtype0*
_output_shapes
: ≤
TensorListPushBack_12TensorListPushBack9tensorlistpushback_12_matmul_readvariableop_0_accumulatorMatMul/ReadVariableOp:value:0*
element_dtype0*
_output_shapes
: Ќ
TensorListPushBack_13TensorListPushBackGtensorlistpushback_13_tensorarrayv2read_tensorlistgetitem_0_accumulator*TensorArrayV2Read/TensorListGetItem:item:0*
element_dtype0*
_output_shapes
: ґ
TensorListPushBack_14TensorListPushBack;tensorlistpushback_14_matmul_1_readvariableop_0_accumulatorMatMul_1/ReadVariableOp:value:0*
element_dtype0*
_output_shapes
: Ъ
TensorListPushBack_15TensorListPushBack1tensorlistpushback_15_placeholder_2_0_accumulatorplaceholder_2*
element_dtype0*
_output_shapes
: "H
!biasadd_readvariableop_resource_0#biasadd_readvariableop_resource_0_0"8
tensorlistpushback"TensorListPushBack:output_handle:0"†
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Otensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0_0">
tensorlistpushback_10%TensorListPushBack_10:output_handle:0"<
tensorlistpushback_1$TensorListPushBack_1:output_handle:0"<
tensorlistpushback_2$TensorListPushBack_2:output_handle:0">
tensorlistpushback_11%TensorListPushBack_11:output_handle:0">
tensorlistpushback_12%TensorListPushBack_12:output_handle:0"J
"matmul_1_readvariableop_resource_0$matmul_1_readvariableop_resource_0_0"<
tensorlistpushback_3$TensorListPushBack_3:output_handle:0">
tensorlistpushback_13%TensorListPushBack_13:output_handle:0"<
tensorlistpushback_4$TensorListPushBack_4:output_handle:0">
tensorlistpushback_14%TensorListPushBack_14:output_handle:0"<
tensorlistpushback_5$TensorListPushBack_5:output_handle:0">
tensorlistpushback_15%TensorListPushBack_15:output_handle:0"<
tensorlistpushback_6$TensorListPushBack_6:output_handle:0"<
tensorlistpushback_7$TensorListPushBack_7:output_handle:0"<
tensorlistpushback_8$TensorListPushBack_8:output_handle:0"<
tensorlistpushback_9$TensorListPushBack_9:output_handle:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"$
strided_slice_0strided_slice_0_0"F
 matmul_readvariableop_resource_0"matmul_readvariableop_resource_0_0*q
_input_shapes`
^: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : : : : : : : : : : : : 22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :	 : : : : :  : : : : : : : : : : : : : : : :
 
З
™
)__inference_restored_function_body_356945

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-349318*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_349317*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*W
_output_shapesE
C:€€€€€€€€€:€€€€€€€€€::€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
Ѓ
и
C__inference_lstm_19_layer_call_and_return_conditional_losses_344737

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИҐStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ў
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-344704*)
f$R"
 __inference_standard_lstm_344703*
Tout	
2**
config_proto

CPU

GPU 2J 8*
Tin

2*o
_output_shapes]
[:€€€€€€€€€:€€€€€€€€€€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
нG
Ц
 __inference_standard_lstm_344703

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
_lower_using_switch_merge(*
parallel_iterations *0
cond(R&
$__inference_while_cond_317942_344616*
_num_original_outputs*0
body(R&
$__inference_while_body_317943_344448*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ≥
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€є

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ґ

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€ґ

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*e
_input_shapesT
R:€€€€€€€€€€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
ґ
ќ
H__inference_sequential_9_layer_call_and_return_conditional_losses_359537
lstm_18_input*
&lstm_18_statefulpartitionedcall_args_1*
&lstm_18_statefulpartitionedcall_args_2*
&lstm_18_statefulpartitionedcall_args_3*
&lstm_19_statefulpartitionedcall_args_1*
&lstm_19_statefulpartitionedcall_args_2*
&lstm_19_statefulpartitionedcall_args_3*
&dense_9_statefulpartitionedcall_args_1*
&dense_9_statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallҐdense_9/StatefulPartitionedCallҐlstm_18/StatefulPartitionedCallҐlstm_19/StatefulPartitionedCallЭ
lstm_18/StatefulPartitionedCallStatefulPartitionedCalllstm_18_input&lstm_18_statefulpartitionedcall_args_1&lstm_18_statefulpartitionedcall_args_2&lstm_18_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-356868*2
f-R+
)__inference_restored_function_body_356867*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:€€€€€€€€€
і
lstm_19/StatefulPartitionedCallStatefulPartitionedCall(lstm_18/StatefulPartitionedCall:output:0&lstm_19_statefulpartitionedcall_args_1&lstm_19_statefulpartitionedcall_args_2&lstm_19_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-356909*2
f-R+
)__inference_restored_function_body_356908*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€Л
dense_9/StatefulPartitionedCallStatefulPartitionedCall(lstm_19/StatefulPartitionedCall:output:0&dense_9_statefulpartitionedcall_args_1&dense_9_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-356946*2
f-R+
)__inference_restored_function_body_356945*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€ј
StatefulPartitionedCallStatefulPartitionedCall&dense_9_statefulpartitionedcall_args_1 ^dense_9/StatefulPartitionedCall*-
_gradient_op_typePartitionedCall-356980*2
f-R+
)__inference_restored_function_body_356979*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*
_output_shapes
: р
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^StatefulPartitionedCall ^dense_9/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*J
_input_shapes9
7:€€€€€€€€€
::::::::2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :- )
'
_user_specified_namelstm_18_input: : : 
нG
Ц
 __inference_standard_lstm_349574

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
_lower_using_switch_merge(*
parallel_iterations *0
cond(R&
$__inference_while_cond_322476_348907*
_num_original_outputs*0
body(R&
$__inference_while_body_322477_349487*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ≥
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€є

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ґ

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€ґ

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*e
_input_shapesT
R:€€€€€€€€€€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
™
ќ
)__inference_restored_function_body_356924

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-350761*L
fGRE
C__inference_lstm_19_layer_call_and_return_conditional_losses_350760*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
ґ
ќ
H__inference_sequential_9_layer_call_and_return_conditional_losses_359555
lstm_18_input*
&lstm_18_statefulpartitionedcall_args_1*
&lstm_18_statefulpartitionedcall_args_2*
&lstm_18_statefulpartitionedcall_args_3*
&lstm_19_statefulpartitionedcall_args_1*
&lstm_19_statefulpartitionedcall_args_2*
&lstm_19_statefulpartitionedcall_args_3*
&dense_9_statefulpartitionedcall_args_1*
&dense_9_statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallҐdense_9/StatefulPartitionedCallҐlstm_18/StatefulPartitionedCallҐlstm_19/StatefulPartitionedCallЭ
lstm_18/StatefulPartitionedCallStatefulPartitionedCalllstm_18_input&lstm_18_statefulpartitionedcall_args_1&lstm_18_statefulpartitionedcall_args_2&lstm_18_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-356884*2
f-R+
)__inference_restored_function_body_356883*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:€€€€€€€€€
і
lstm_19/StatefulPartitionedCallStatefulPartitionedCall(lstm_18/StatefulPartitionedCall:output:0&lstm_19_statefulpartitionedcall_args_1&lstm_19_statefulpartitionedcall_args_2&lstm_19_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-356925*2
f-R+
)__inference_restored_function_body_356924*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€Л
dense_9/StatefulPartitionedCallStatefulPartitionedCall(lstm_19/StatefulPartitionedCall:output:0&dense_9_statefulpartitionedcall_args_1&dense_9_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-356946*2
f-R+
)__inference_restored_function_body_356945*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:€€€€€€€€€ј
StatefulPartitionedCallStatefulPartitionedCall&dense_9_statefulpartitionedcall_args_1 ^dense_9/StatefulPartitionedCall*-
_gradient_op_typePartitionedCall-356980*2
f-R+
)__inference_restored_function_body_356979*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*
_output_shapes
: р
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^StatefulPartitionedCall ^dense_9/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*J
_input_shapes9
7:€€€€€€€€€
::::::::2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :- )
'
_user_specified_namelstm_18_input: : : 
”

Н
-__inference_sequential_9_layer_call_fn_353889
lstm_18_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИҐStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-353862*Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_353861*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2	*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*J
_input_shapes9
7:€€€€€€€€€
::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :- )
'
_user_specified_namelstm_18_input: : : 
№+
В
$__inference_while_body_325828_350639
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€§
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:PН
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P®
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pt
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PҐ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
№+
В
$__inference_while_body_322477_349487
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€§
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:PН
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P®
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pt
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PҐ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
м
§
$__inference_while_cond_322947_346848
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : :  : : :	 : :
 
ЈG
Ц
 __inference_standard_lstm_353803

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
€€€€€€€€€B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
_lower_using_switch_merge(*
parallel_iterations *0
cond(R&
$__inference_while_cond_318896_353716*
_num_original_outputs*0
body(R&
$__inference_while_body_318897_344826*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€
[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ≥
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€∞

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:€€€€€€€€€
ґ

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€ґ

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*\
_input_shapesK
I:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
Ы
и
C__inference_lstm_18_layer_call_and_return_conditional_losses_350214

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИҐStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ћ
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-350181*)
f$R"
 __inference_standard_lstm_350180*
Tout	
2**
config_proto

CPU

GPU 2J 8*
Tin

2*а
_output_shapesЌ
 :€€€€€€€€€:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€: :
€€€€€€€€€::::: : : : : : : : : : : : : : : : : : : : : : :
€€€€€€€€€:Ж
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€
"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
м
§
$__inference_while_cond_322476_348907
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : :  : : :	 : :
 
нG
Ц
 __inference_standard_lstm_354571

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
_lower_using_switch_merge(*
parallel_iterations *0
cond(R&
$__inference_while_cond_317460_354484*
_num_original_outputs*0
body(R&
$__inference_while_body_317461_344600*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ≥
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€є

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ґ

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€ґ

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*e
_input_shapesT
R:€€€€€€€€€€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
Щ
§
$__inference_while_cond_319391_344764
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : :  : : :	 : :
 
ґ
к
C__inference_lstm_19_layer_call_and_return_conditional_losses_354468
inputs_0"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИҐStatefulPartitionedCall=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :и*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€џ
StatefulPartitionedCallStatefulPartitionedCallinputs_0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-354435*)
f$R"
 __inference_standard_lstm_354434*
Tout	
2**
config_proto

CPU

GPU 2J 8*
Tin

2*o
_output_shapes]
[:€€€€€€€€€:€€€€€€€€€€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall: :( $
"
_user_specified_name
inputs/0: : 
ЧS
К
.__inference_while_body_323435_343993_rewritten
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0_0S
Otensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0_0&
"matmul_readvariableop_resource_0_0(
$matmul_1_readvariableop_resource_0_0'
#biasadd_readvariableop_resource_0_0*
&tensorlistpushback_mul_2_0_accumulator2
.tensorlistpushback_1_placeholder_0_accumulator0
,tensorlistpushback_2_sigmoid_2_0_accumulator-
)tensorlistpushback_3_tanh_1_0_accumulator*
&tensorlistpushback_4_mul_0_accumulator,
(tensorlistpushback_5_mul_1_0_accumulator0
,tensorlistpushback_6_sigmoid_1_0_accumulator4
0tensorlistpushback_7_placeholder_3_0_accumulator.
*tensorlistpushback_8_sigmoid_0_accumulator+
'tensorlistpushback_9_tanh_0_accumulator.
*tensorlistpushback_10_matmul_0_accumulator0
,tensorlistpushback_11_matmul_1_0_accumulator=
9tensorlistpushback_12_matmul_readvariableop_0_accumulatorK
Gtensorlistpushback_13_tensorarrayv2read_tensorlistgetitem_0_accumulator?
;tensorlistpushback_14_matmul_1_readvariableop_0_accumulator5
1tensorlistpushback_15_placeholder_2_0_accumulator
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
tensorlistpushback
tensorlistpushback_1
tensorlistpushback_2
tensorlistpushback_3
tensorlistpushback_4
tensorlistpushback_5
tensorlistpushback_6
tensorlistpushback_7
tensorlistpushback_8
tensorlistpushback_9
tensorlistpushback_10
tensorlistpushback_11
tensorlistpushback_12
tensorlistpushback_13
tensorlistpushback_14
tensorlistpushback_15ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Р
#TensorArrayV2Read/TensorListGetItemTensorListGetItemOtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:€€€€€€€€€†
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_resource_0_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Ц
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€§
MatMul_1/ReadVariableOpReadVariableOp$matmul_1_readvariableop_resource_0_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:}
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€m
addAddV2MatMul:product:0MatMul_1:product:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Ґ
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_resource_0_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: џ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*Д
_output_shapesr
p:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€]
SigmoidSigmoidsplit:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€_
	Sigmoid_1Sigmoidsplit:output:1*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€W
TanhTanhsplit:output:2*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€_
	Sigmoid_2Sigmoidsplit:output:3*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: Л
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ь

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Н

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ю

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ю

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€И
TensorListPushBackTensorListPushBack&tensorlistpushback_mul_2_0_accumulator	mul_2:z:0*
element_dtype0*
_output_shapes
: Ф
TensorListPushBack_1TensorListPushBack.tensorlistpushback_1_placeholder_0_accumulatorplaceholder*
element_dtype0*
_output_shapes
: Ф
TensorListPushBack_2TensorListPushBack,tensorlistpushback_2_sigmoid_2_0_accumulatorSigmoid_2:y:0*
element_dtype0*
_output_shapes
: О
TensorListPushBack_3TensorListPushBack)tensorlistpushback_3_tanh_1_0_accumulator
Tanh_1:y:0*
element_dtype0*
_output_shapes
: И
TensorListPushBack_4TensorListPushBack&tensorlistpushback_4_mul_0_accumulatormul:z:0*
element_dtype0*
_output_shapes
: М
TensorListPushBack_5TensorListPushBack(tensorlistpushback_5_mul_1_0_accumulator	mul_1:z:0*
element_dtype0*
_output_shapes
: Ф
TensorListPushBack_6TensorListPushBack,tensorlistpushback_6_sigmoid_1_0_accumulatorSigmoid_1:y:0*
element_dtype0*
_output_shapes
: Ш
TensorListPushBack_7TensorListPushBack0tensorlistpushback_7_placeholder_3_0_accumulatorplaceholder_3*
element_dtype0*
_output_shapes
: Р
TensorListPushBack_8TensorListPushBack*tensorlistpushback_8_sigmoid_0_accumulatorSigmoid:y:0*
element_dtype0*
_output_shapes
: К
TensorListPushBack_9TensorListPushBack'tensorlistpushback_9_tanh_0_accumulatorTanh:y:0*
element_dtype0*
_output_shapes
: Ц
TensorListPushBack_10TensorListPushBack*tensorlistpushback_10_matmul_0_accumulatorMatMul:product:0*
element_dtype0*
_output_shapes
: Ъ
TensorListPushBack_11TensorListPushBack,tensorlistpushback_11_matmul_1_0_accumulatorMatMul_1:product:0*
element_dtype0*
_output_shapes
: ≤
TensorListPushBack_12TensorListPushBack9tensorlistpushback_12_matmul_readvariableop_0_accumulatorMatMul/ReadVariableOp:value:0*
element_dtype0*
_output_shapes
: Ќ
TensorListPushBack_13TensorListPushBackGtensorlistpushback_13_tensorarrayv2read_tensorlistgetitem_0_accumulator*TensorArrayV2Read/TensorListGetItem:item:0*
element_dtype0*
_output_shapes
: ґ
TensorListPushBack_14TensorListPushBack;tensorlistpushback_14_matmul_1_readvariableop_0_accumulatorMatMul_1/ReadVariableOp:value:0*
element_dtype0*
_output_shapes
: Ъ
TensorListPushBack_15TensorListPushBack1tensorlistpushback_15_placeholder_2_0_accumulatorplaceholder_2*
element_dtype0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"$
strided_slice_0strided_slice_0_0"F
 matmul_readvariableop_resource_0"matmul_readvariableop_resource_0_0"H
!biasadd_readvariableop_resource_0#biasadd_readvariableop_resource_0_0"8
tensorlistpushback"TensorListPushBack:output_handle:0"†
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Otensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0_0">
tensorlistpushback_10%TensorListPushBack_10:output_handle:0"<
tensorlistpushback_1$TensorListPushBack_1:output_handle:0"<
tensorlistpushback_2$TensorListPushBack_2:output_handle:0">
tensorlistpushback_11%TensorListPushBack_11:output_handle:0">
tensorlistpushback_12%TensorListPushBack_12:output_handle:0"J
"matmul_1_readvariableop_resource_0$matmul_1_readvariableop_resource_0_0"<
tensorlistpushback_3$TensorListPushBack_3:output_handle:0">
tensorlistpushback_13%TensorListPushBack_13:output_handle:0"<
tensorlistpushback_4$TensorListPushBack_4:output_handle:0">
tensorlistpushback_14%TensorListPushBack_14:output_handle:0"<
tensorlistpushback_5$TensorListPushBack_5:output_handle:0">
tensorlistpushback_15%TensorListPushBack_15:output_handle:0"<
tensorlistpushback_6$TensorListPushBack_6:output_handle:0"<
tensorlistpushback_7$TensorListPushBack_7:output_handle:0"<
tensorlistpushback_8$TensorListPushBack_8:output_handle:0"<
tensorlistpushback_9$TensorListPushBack_9:output_handle:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*q
_input_shapes`
^: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : : : : : : : : : : : : 22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :
 : : : : : :	 : : : : :  : : : : : : : : : : 
ь
М
C__inference_dense_9_layer_call_and_return_conditional_losses_347646

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ-dense_9/kernel/Regularizer/Abs/ReadVariableOpҐ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€“
-dense_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Е
dense_9/kernel/Regularizer/AbsAbs5dense_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:q
 dense_9/kernel/Regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:Х
dense_9/kernel/Regularizer/SumSum"dense_9/kernel/Regularizer/Abs:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/mul/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: Ъ
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: Ч
dense_9/kernel/Regularizer/addAddV2)dense_9/kernel/Regularizer/add/x:output:0"dense_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_9/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€::2^
-dense_9/kernel/Regularizer/Abs/ReadVariableOp-dense_9/kernel/Regularizer/Abs/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
К
љ
A__forward_dense_9_layer_call_and_return_conditional_losses_357219
inputs_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
relu
matmul_readvariableop

inputsИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ-dense_9/kernel/Regularizer/Abs/ReadVariableOpҐ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:k
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€“
-dense_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Е
dense_9/kernel/Regularizer/AbsAbs5dense_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:q
 dense_9/kernel/Regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:Х
dense_9/kernel/Regularizer/SumSum"dense_9/kernel/Regularizer/Abs:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/mul/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: Ъ
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: Ч
dense_9/kernel/Regularizer/addAddV2)dense_9/kernel/Regularizer/add/x:output:0"dense_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_9/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"
reluRelu:activations:0"
identityIdentity:output:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0"
inputsinputs_0*q
backward_function_nameWU__inference___backward_dense_9_layer_call_and_return_conditional_losses_357205_357220*.
_input_shapes
:€€€€€€€€€::2^
-dense_9/kernel/Regularizer/Abs/ReadVariableOp-dense_9/kernel/Regularizer/Abs/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
Щ
§
$__inference_while_cond_323907_344399
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : :  : : :	 : :
 
нG
Ц
 __inference_standard_lstm_352964

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:Ќ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€К
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:P{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PЦ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Pm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Pd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€PЕ
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Pm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€PG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:£
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
_lower_using_switch_merge(*
parallel_iterations *0
cond(R&
$__inference_while_cond_316034_352877*
_num_original_outputs*0
body(R&
$__inference_while_body_316035_352861*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€h
strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: ≥
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€є

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ґ

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€ґ

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*e
_input_shapesT
R:€€€€€€€€€€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namekernel:&"
 
_user_specified_nameinit_c
Щ
§
$__inference_while_cond_325354_344844
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :€€€€€€€€€:€€€€€€€€€: : :::: : : : : :  : : :	 : :
 "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ї
serving_default¶
K
lstm_18_input:
serving_default_lstm_18_input:0€€€€€€€€€
;
dense_90
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:Гє
л+
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer

signatures
trainable_variables
	variables
	regularization_losses

	keras_api
*R&call_and_return_all_conditional_losses
S_default_save_signature
T__call__"Ю)
_tf_keras_sequential€({"class_name": "Sequential", "name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_9", "layers": [{"class_name": "LSTM", "config": {"name": "lstm_18", "trainable": true, "batch_input_shape": [null, 10, 7], "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTM", "config": {"name": "lstm_19", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 7], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "LSTM", "config": {"name": "lstm_18", "trainable": true, "batch_input_shape": [null, 10, 7], "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTM", "config": {"name": "lstm_19", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": [], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
µ
trainable_variables
	variables
regularization_losses
	keras_api
*U&call_and_return_all_conditional_losses
V__call__"¶
_tf_keras_layerМ{"class_name": "InputLayer", "name": "lstm_18_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 10, 7], "config": {"batch_input_shape": [null, 10, 7], "dtype": "float32", "sparse": false, "name": "lstm_18_input"}}
љ

cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
*W&call_and_return_all_conditional_losses
X__call__"Ф	
_tf_keras_layerъ{"class_name": "LSTM", "name": "lstm_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 10, 7], "config": {"name": "lstm_18", "trainable": true, "batch_input_shape": [null, 10, 7], "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 7], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
Т

cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
*Y&call_and_return_all_conditional_losses
Z__call__"й
_tf_keras_layerѕ{"class_name": "LSTM", "name": "lstm_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_19", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 20], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
µ

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
*[&call_and_return_all_conditional_losses
\__call__"Р
_tf_keras_layerц{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}}
г
!iter

"beta_1

#beta_2
	$decay
%learning_ratemBmC&mD'mE(mF)mG*mH+mIvJvK&vL'vM(vN)vO*vP+vQ"
	optimizer
,
]serving_default"
signature_map
X
&0
'1
(2
)3
*4
+5
6
7"
trackable_list_wrapper
X
&0
'1
(2
)3
*4
+5
6
7"
trackable_list_wrapper
'
^0"
trackable_list_wrapper
Ј
trainable_variables
,metrics
	variables
-layer_regularization_losses

.layers
/non_trainable_variables
	regularization_losses
T__call__
S_default_save_signature
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
trainable_variables
0metrics
	variables
1layer_regularization_losses

2layers
3non_trainable_variables
regularization_losses
V__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
ч

&kernel
'recurrent_kernel
(bias
4	keras_api"є
_tf_keras_layerЯ{"class_name": "LSTMCell", "name": "lstm_cell_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_cell_18", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
trainable_variables
5metrics
	variables
6layer_regularization_losses

7layers
8non_trainable_variables
regularization_losses
X__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
ч

)kernel
*recurrent_kernel
+bias
9	keras_api"є
_tf_keras_layerЯ{"class_name": "LSTMCell", "name": "lstm_cell_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_cell_19", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
trainable_variables
:metrics
	variables
;layer_regularization_losses

<layers
=non_trainable_variables
regularization_losses
Z__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
 :2dense_9/kernel
:2dense_9/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
^0"
trackable_list_wrapper
Ъ
trainable_variables
>metrics
	variables
?layer_regularization_losses

@layers
Anon_trainable_variables
regularization_losses
\__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 :P2lstm_18/kernel
*:(P2lstm_18/recurrent_kernel
:P2lstm_18/bias
 :P2lstm_19/kernel
*:(P2lstm_19/recurrent_kernel
:P2lstm_19/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
^0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
%:#2Adam/dense_9/kernel/m
:2Adam/dense_9/bias/m
%:#P2Adam/lstm_18/kernel/m
/:-P2Adam/lstm_18/recurrent_kernel/m
:P2Adam/lstm_18/bias/m
%:#P2Adam/lstm_19/kernel/m
/:-P2Adam/lstm_19/recurrent_kernel/m
:P2Adam/lstm_19/bias/m
%:#2Adam/dense_9/kernel/v
:2Adam/dense_9/bias/v
%:#P2Adam/lstm_18/kernel/v
/:-P2Adam/lstm_18/recurrent_kernel/v
:P2Adam/lstm_18/bias/v
%:#P2Adam/lstm_19/kernel/v
/:-P2Adam/lstm_19/recurrent_kernel/v
:P2Adam/lstm_19/bias/v
Џ2„
H__inference_sequential_9_layer_call_and_return_conditional_losses_359555
H__inference_sequential_9_layer_call_and_return_conditional_losses_359537ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
й2ж
!__inference__wrapped_model_359518ј
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *0Ґ-
+К(
lstm_18_input€€€€€€€€€

Ъ2Ч
-__inference_sequential_9_layer_call_fn_359613
-__inference_sequential_9_layer_call_fn_359584ґ
ѓ≤Ђ
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
е2в
C__inference_lstm_18_layer_call_and_return_conditional_losses_346969
C__inference_lstm_18_layer_call_and_return_conditional_losses_349608
C__inference_lstm_18_layer_call_and_return_conditional_losses_354092
C__inference_lstm_18_layer_call_and_return_conditional_losses_350214Ћ
ƒ≤ј
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
щ2ц
(__inference_lstm_18_layer_call_fn_353898
(__inference_lstm_18_layer_call_fn_353007
(__inference_lstm_18_layer_call_fn_356060
(__inference_lstm_18_layer_call_fn_347457Ћ
ƒ≤ј
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
е2в
C__inference_lstm_19_layer_call_and_return_conditional_losses_354468
C__inference_lstm_19_layer_call_and_return_conditional_losses_346410
C__inference_lstm_19_layer_call_and_return_conditional_losses_350384
C__inference_lstm_19_layer_call_and_return_conditional_losses_350760Ћ
ƒ≤ј
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
щ2ц
(__inference_lstm_19_layer_call_fn_354614
(__inference_lstm_19_layer_call_fn_349756
(__inference_lstm_19_layer_call_fn_345332
(__inference_lstm_19_layer_call_fn_344746Ћ
ƒ≤ј
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
г2а
C__inference_dense_9_layer_call_and_return_conditional_losses_349317Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
»2≈
(__inference_dense_9_layer_call_fn_347654Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
9B7
$__inference_signature_wrapper_359633lstm_18_input
≥2∞
__inference_loss_fn_0_344777П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ Ѕ
H__inference_sequential_9_layer_call_and_return_conditional_losses_359555u&'()*+BҐ?
8Ґ5
+К(
lstm_18_input€€€€€€€€€

p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ѕ
H__inference_sequential_9_layer_call_and_return_conditional_losses_359537u&'()*+BҐ?
8Ґ5
+К(
lstm_18_input€€€€€€€€€

p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ю
!__inference__wrapped_model_359518y&'()*+:Ґ7
0Ґ-
+К(
lstm_18_input€€€€€€€€€

™ "1™.
,
dense_9!К
dense_9€€€€€€€€€Щ
-__inference_sequential_9_layer_call_fn_359613h&'()*+BҐ?
8Ґ5
+К(
lstm_18_input€€€€€€€€€

p 

 
™ "К€€€€€€€€€Щ
-__inference_sequential_9_layer_call_fn_359584h&'()*+BҐ?
8Ґ5
+К(
lstm_18_input€€€€€€€€€

p

 
™ "К€€€€€€€€€“
C__inference_lstm_18_layer_call_and_return_conditional_losses_346969К&'(OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ “
C__inference_lstm_18_layer_call_and_return_conditional_losses_349608К&'(OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ Є
C__inference_lstm_18_layer_call_and_return_conditional_losses_354092q&'(?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€


 
p 

 
™ ")Ґ&
К
0€€€€€€€€€

Ъ Є
C__inference_lstm_18_layer_call_and_return_conditional_losses_350214q&'(?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€


 
p

 
™ ")Ґ&
К
0€€€€€€€€€

Ъ Р
(__inference_lstm_18_layer_call_fn_353898d&'(?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€


 
p 

 
™ "К€€€€€€€€€
©
(__inference_lstm_18_layer_call_fn_353007}&'(OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "%К"€€€€€€€€€€€€€€€€€€©
(__inference_lstm_18_layer_call_fn_356060}&'(OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "%К"€€€€€€€€€€€€€€€€€€Р
(__inference_lstm_18_layer_call_fn_347457d&'(?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€


 
p

 
™ "К€€€€€€€€€
ƒ
C__inference_lstm_19_layer_call_and_return_conditional_losses_354468})*+OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ і
C__inference_lstm_19_layer_call_and_return_conditional_losses_346410m)*+?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€


 
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ƒ
C__inference_lstm_19_layer_call_and_return_conditional_losses_350384})*+OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ і
C__inference_lstm_19_layer_call_and_return_conditional_losses_350760m)*+?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€


 
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ь
(__inference_lstm_19_layer_call_fn_354614p)*+OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "К€€€€€€€€€М
(__inference_lstm_19_layer_call_fn_349756`)*+?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€


 
p 

 
™ "К€€€€€€€€€М
(__inference_lstm_19_layer_call_fn_345332`)*+?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€


 
p

 
™ "К€€€€€€€€€Ь
(__inference_lstm_19_layer_call_fn_344746p)*+OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€£
C__inference_dense_9_layer_call_and_return_conditional_losses_349317\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ {
(__inference_dense_9_layer_call_fn_347654O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€≥
$__inference_signature_wrapper_359633К&'()*+KҐH
Ґ 
A™>
<
lstm_18_input+К(
lstm_18_input€€€€€€€€€
"1™.
,
dense_9!К
dense_9€€€€€€€€€;
__inference_loss_fn_0_344777Ґ

Ґ 
™ "К 