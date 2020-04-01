Search.setIndex({docnames:["builtins","compiler","constant","dimension","download","equation","finite-difference","function","grid","grids","index","operator","precsparsefunction","precsparsetimefunction","sparsefunction","sparsetimefunction","subdomain","symbolic","tensorfunction","tensortimefunction","timefunction","tutorials","userapi","vectorfunction","vectortimefunction"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.index":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["builtins.rst","compiler.rst","constant.rst","dimension.rst","download.rst","equation.rst","finite-difference.rst","function.rst","grid.rst","grids.rst","index.rst","operator.rst","precsparsefunction.rst","precsparsetimefunction.rst","sparsefunction.rst","sparsetimefunction.rst","subdomain.rst","symbolic.rst","tensorfunction.rst","tensortimefunction.rst","timefunction.rst","tutorials.rst","userapi.rst","vectorfunction.rst","vectortimefunction.rst"],objects:{"devito.builtins":{assign:[0,1,1,""],gaussian_smooth:[0,1,1,""],initialize_function:[0,1,1,""],inner:[0,1,1,""],mmax:[0,1,1,""],mmin:[0,1,1,""],norm:[0,1,1,""],smooth:[0,1,1,""],sumall:[0,1,1,""]},"devito.finite_differences":{finite_difference:[6,0,0,"-"]},"devito.finite_differences.finite_difference":{cross_derivative:[6,1,1,""],first_derivative:[6,1,1,""],generic_derivative:[6,1,1,""],second_derivative:[6,1,1,""]},"devito.operator":{operator:[11,0,0,"-"]},"devito.operator.operator":{Operator:[11,2,1,""]},"devito.operator.operator.Operator":{apply:[11,3,1,""],arguments:[11,3,1,""],cfunction:[11,3,1,""],dimensions:[11,4,1,""],input:[11,4,1,""],objects:[11,4,1,""],output:[11,4,1,""]},"devito.types":{Constant:[2,2,1,""],Function:[7,2,1,""],Grid:[8,2,1,""],PrecomputedSparseFunction:[12,2,1,""],PrecomputedSparseTimeFunction:[13,2,1,""],SparseFunction:[14,2,1,""],SparseTimeFunction:[15,2,1,""],SubDomain:[16,2,1,""],TensorFunction:[18,2,1,""],TensorTimeFunction:[19,2,1,""],TimeFunction:[20,2,1,""],VectorFunction:[23,2,1,""],VectorTimeFunction:[24,2,1,""],dimension:[3,0,0,"-"],equation:[5,0,0,"-"]},"devito.types.Constant":{data:[2,3,1,""],is_const:[2,3,1,""]},"devito.types.Function":{avg:[7,3,1,""],data:[7,3,1,""],data_domain:[7,3,1,""],data_ro_domain:[7,3,1,""],data_ro_with_halo:[7,3,1,""],data_with_halo:[7,3,1,""],dimensions:[7,3,1,""],dtype:[7,3,1,""],grid:[7,3,1,""],name:[7,3,1,""],shape:[7,4,1,""],shape_allocated:[7,4,1,""],shape_global:[7,4,1,""],shape_with_halo:[7,4,1,""],space_dimensions:[7,4,1,""],space_order:[7,3,1,""],sum:[7,3,1,""]},"devito.types.Grid":{comm:[8,3,1,""],dim:[8,3,1,""],dimension_map:[8,3,1,""],dimensions:[8,3,1,""],distributor:[8,3,1,""],dtype:[8,3,1,""],extent:[8,3,1,""],interior:[8,3,1,""],is_distributed:[8,3,1,""],origin:[8,3,1,""],origin_offset:[8,3,1,""],shape:[8,3,1,""],shape_local:[8,3,1,""],spacing:[8,3,1,""],spacing_map:[8,3,1,""],spacing_symbols:[8,3,1,""],stepping_dim:[8,3,1,""],subdomains:[8,3,1,""],time_dim:[8,3,1,""],volume_cell:[8,3,1,""]},"devito.types.PrecomputedSparseFunction":{data:[12,3,1,""],data_domain:[12,3,1,""],data_ro_domain:[12,3,1,""],data_ro_with_halo:[12,3,1,""],data_with_halo:[12,3,1,""],dimensions:[12,3,1,""],dtype:[12,3,1,""],grid:[12,3,1,""],gridpoints:[12,3,1,""],inject:[12,3,1,""],interpolate:[12,3,1,""],name:[12,3,1,""],shape:[12,4,1,""],space_order:[12,3,1,""]},"devito.types.PrecomputedSparseTimeFunction":{data:[13,3,1,""],data_domain:[13,3,1,""],data_ro_domain:[13,3,1,""],data_ro_with_halo:[13,3,1,""],data_with_halo:[13,3,1,""],dimensions:[13,3,1,""],dtype:[13,3,1,""],grid:[13,3,1,""],gridpoints:[13,3,1,""],inject:[13,3,1,""],interpolate:[13,3,1,""],name:[13,3,1,""],shape:[13,4,1,""],space_order:[13,3,1,""],time_order:[13,3,1,""]},"devito.types.SparseFunction":{data:[14,3,1,""],data_domain:[14,3,1,""],data_ro_domain:[14,3,1,""],data_ro_with_halo:[14,3,1,""],data_with_halo:[14,3,1,""],dimensions:[14,3,1,""],dtype:[14,3,1,""],grid:[14,3,1,""],gridpoints:[14,3,1,""],guard:[14,3,1,""],inject:[14,3,1,""],interpolate:[14,3,1,""],name:[14,3,1,""],shape:[14,4,1,""],space_order:[14,3,1,""]},"devito.types.SparseTimeFunction":{data:[15,3,1,""],data_domain:[15,3,1,""],data_ro_domain:[15,3,1,""],data_ro_with_halo:[15,3,1,""],data_with_halo:[15,3,1,""],dimensions:[15,3,1,""],dtype:[15,3,1,""],grid:[15,3,1,""],gridpoints:[15,3,1,""],guard:[15,3,1,""],inject:[15,3,1,""],interpolate:[15,3,1,""],name:[15,3,1,""],shape:[15,4,1,""],space_order:[15,3,1,""],time_order:[15,3,1,""]},"devito.types.SubDomain":{define:[16,3,1,""],name:[16,4,1,""]},"devito.types.TensorFunction":{shape:[18,3,1,""]},"devito.types.TensorTimeFunction":{backward:[19,3,1,""],forward:[19,3,1,""],shape:[19,3,1,""]},"devito.types.TimeFunction":{avg:[20,3,1,""],backward:[20,3,1,""],data:[20,3,1,""],data_domain:[20,3,1,""],data_ro_domain:[20,3,1,""],data_ro_with_halo:[20,3,1,""],data_with_halo:[20,3,1,""],dimensions:[20,3,1,""],dtype:[20,3,1,""],forward:[20,3,1,""],grid:[20,3,1,""],name:[20,3,1,""],shape:[20,4,1,""],shape_allocated:[20,4,1,""],shape_global:[20,4,1,""],shape_with_halo:[20,4,1,""],space_dimensions:[20,4,1,""],space_order:[20,3,1,""],sum:[20,3,1,""],time_order:[20,3,1,""]},"devito.types.VectorFunction":{shape:[23,3,1,""]},"devito.types.VectorTimeFunction":{backward:[24,3,1,""],forward:[24,3,1,""],shape:[24,3,1,""]},"devito.types.dimension":{ConditionalDimension:[3,2,1,""],DefaultDimension:[3,2,1,""],Dimension:[3,2,1,""],IncrDimension:[3,2,1,""],ModuloDimension:[3,2,1,""],ShiftedDimension:[3,2,1,""],SpaceDimension:[3,2,1,""],SteppingDimension:[3,2,1,""],SubDimension:[3,2,1,""],TimeDimension:[3,2,1,""]},"devito.types.dimension.ConditionalDimension":{factor:[3,3,1,""],free_symbols:[3,3,1,""],spacing:[3,3,1,""]},"devito.types.dimension.Dimension":{spacing:[3,3,1,""],symbolic_incr:[3,3,1,""],symbolic_max:[3,4,1,""],symbolic_min:[3,4,1,""],symbolic_size:[3,4,1,""]},"devito.types.dimension.SteppingDimension":{symbolic_max:[3,3,1,""],symbolic_min:[3,3,1,""]},"devito.types.equation":{Eq:[5,2,1,""],Inc:[5,2,1,""],solve:[5,1,1,""]},"devito.types.equation.Eq":{evaluate:[5,4,1,""],subdomain:[5,3,1,""],xreplace:[5,3,1,""]},devito:{builtins:[0,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:attribute"},terms:{"case":[3,6,7,8,11,16,20],"class":[2,3,5,7,8,11,12,13,14,15,16,18,19,20,23,24],"default":[0,2,3,5,6,7,8,11,12,13,14,15,18,20],"final":[0,6],"float":[0,2,3,8],"function":[0,3,4,5,6,8,11,14,15,17,18,20,22,23],"import":[0,2,3,4,5,6,7,8,11,14,15,16,18,19,20,23,24],"int":[0,2,3,6,7,8,12,13,14,15,20],"long":4,"new":16,"return":[3,5,6,14,16],"static":3,"true":[2,3,8,13,15,18],"try":[4,5],"while":[3,5,7,11,20],And:3,BCs:21,For:[0,3,4,5,6,7,8,11,16,20],LHS:[0,5],One:20,RHS:[0,5],The:[0,2,3,4,5,6,7,8,11,12,13,14,15,18,19,20,21,23,24],There:[4,14],These:[0,14,15],With:[7,11,12,13,14,15,20],__doc__:[5,7,8,12,13,14,15,16,20],_max:3,_min:3,abc:5,about:[0,10,11,14],abov:[4,16],absorb:0,abstractsparsefunct:[12,14],abstractsparsetimefunct:[13,15],abstracttensor:18,accept:0,access:[3,6,7,11,14,20],accessor:[7,12,13,14,15,20],acm:1,acoust:21,across:[12,13],activ:4,add:11,added:0,addit:[0,7,13,15,20],addition:18,adjoint:6,advanc:11,advantag:[7,12,13,14,15,20],after:[0,4],against:3,algebra:5,alia:[7,12,13,14,15,20],align:[7,14,15,20],all:[0,3,4,7,8,11,12,13,14,15,16,20,21],alloc:[7,12,13,14,15,20],allow:6,along:[0,3,4,5,7,11,15,20],alongsid:4,also:[3,4,6,7,11,20],alter:11,altern:[3,4,20],alwai:[2,7,12,13,14,15,20],among:[3,4],anaconda:4,analog:16,ani:[2,3,4,5,7,8,11,12,13,14,15,20],anoth:3,api:[3,5,10,21],appear:[3,5,11],appli:[5,11,14],applic:11,approach:4,appropri:4,approxim:[7,11,20,22],arbitrari:[3,5,6,13,15],architectur:[1,7,11,12,13,14,15,20],area:[7,12,13,14,15,20],arg:[2,3,7,8,11,12,13,14,15,18,19,20,23,24],argprovid:[2,3,8],argument:[0,2,6,7,11,12,13,14,15,20],around:[5,7,20,21],arrai:[3,11,12,13,14,15],array_lik:0,artifici:3,arxiv:1,ask:4,assign:[0,5,13,15],associ:[3,7,8,12,13,14,15,20],assum:[3,20],assumpt:3,atom:3,augment:5,authent:4,autom:10,automat:[3,7,11,20],avail:[0,1,7,20,21],averag:[0,7,20],avg:[7,20],avoid:20,axi:0,back:[7,12,13,14,15,20],backend:11,backward:[11,19,20,24],barba:21,base:[2,3,5,7,8,12,13,14,15,16,18,19,20,21,23,24],bash:4,basic:[2,3,18,21],basicdimens:3,becom:7,been:[1,4],befor:[4,14,16],being:0,bell:4,below:[5,11,21],benefit:4,between:[3,5,8,14,15],bin:4,blog:21,bool:[3,13,15,18],both:[3,7,20],bound:[3,5,11],boundari:[0,3,7,8,11,13,15,16,20],box:8,branch:3,browser:4,buffer:[3,7,11,12,13,14,15,20],build:3,built:22,builtin:0,cach:3,call:[0,11,14],callabl:[7,12,13,14,15,20],can:[0,2,3,4,5,7,8,10,11,12,13,14,15,16,20,21],cannot:2,capabl:5,captur:7,carri:[2,3,7,11,12,13,14,15,20],cartesian:8,caus:11,caveat:11,cell:8,center:[6,7,11,20],certain:[3,11,14,15],cfd:21,cfunction:11,chang:5,check:[4,10,21],choos:6,claim:[7,12,13,14,15,20],classic:21,clear:[7,14],clone:4,closest:[12,13],code:[0,1,3,4,10,11,14],coeffici:[5,6,12,13],col:[18,19,23,24],collect:14,com:4,comm:8,command:4,commonli:3,commun:8,compact:11,compil:[3,4,10,11],complet:[0,4,11],compon:[4,18],compos:4,compulsori:[7,20],comput:[0,5,7,8,10,11,12,13,14,15,20],concept:21,condit:[3,5,11,14,15],conditionaldimens:3,configur:11,consecut:3,consid:11,constant:[0,5,11,17,22],constitut:[7,12,13,14,15,20],constrain:[3,11],construct:[0,5],contain:[0,4,7,11,12,13,20],content:3,context:[7,12,13,14,15,20],contigu:[3,16],continu:4,contract:[5,11],contribut:[1,4,12,13],control:[7,12,13,14,15,20],conveni:0,convent:8,convert:[14,15],convex:3,coordin:[0,8,11,12,13,14,15],coordinates_data:[14,15],copi:4,core:[5,21],corner:[7,11,20],correct:8,correspond:[0,12,13,14,15],coupl:21,cover:21,creat:[0,2,3,4,7,8,11,12,13,14,15,16,20],creation:[7,14,15,20],cross:6,cross_deriv:6,ctrl:4,ctype:11,current:1,cyclic:[3,20],d_m:11,d_size:16,d_sizem:16,damp:0,dask:21,data:[0,2,7,8,11,12,13,14,15,20],data_domain:[7,12,13,14,15,20],data_ro:[7,12,13,14,15,20],data_ro_domain:[7,12,13,14,15,20],data_ro_with_halo:[7,12,13,14,15,20],data_with_halo:[7,12,13,14,15,20],datasymbol:[2,3],decompos:[7,20],decomposit:[3,8],dedic:3,def:16,default_alloc:[7,12,13,14,15,20],default_valu:3,defaultdimens:3,defin:[0,3,5,7,8,10,11,12,13,14,15,16,20],definit:10,dens:[7,11,20],depend:[4,7,8,14,20],deprec:[7,20],deriv:[3,5,6,7,12,13,14,15,20],deriv_ord:6,deriveddimens:3,describ:[1,16],detail:10,detect:11,dev:4,develop:[0,4],deviat:0,devito:[0,2,3,5,6,7,8,11,12,13,14,15,16,18,19,20,21,23,24],devitocod:4,diagon:18,dict:[0,5,6,11],dictionari:0,differ:[0,3,5,7,10,11,14,20,21,22],differenti:[5,6,7,18],dim:[6,7,8,20],dimens:[0,2,5,6,7,8,11,12,13,14,15,16,17,18,19,20,22,23,24],dimension:[0,7,12,13,14,15,16,20],dimension_map:8,direct:[6,11],directli:[2,3],dirichlet:21,discov:11,discret:[6,7,8,12,13,14,15,20,21],discretefunct:[7,18],discretis:[7,12,13,14,15,20],distanc:3,distribut:[3,8,11,21],distributor:8,divisor:3,document:[10,11],doe:[0,5,8],doesn:5,doing:[4,5],domain:[0,3,5,7,8,11,12,13,14,15,16,20],don:[0,4],dot:0,down:11,download:[4,10],drop:[3,20],dtype:[0,2,7,8,12,13,14,15,20],dummi:3,dure:14,dx2:[6,7],dxdy:6,dxl:7,dxr:7,each:[0,8,12,13,14,15,16,18],easili:11,either:[4,5,11],elast:21,element:[7,12,13,14,15,20],emploi:[3,10],empti:[7,20],enabl:20,encapsul:[7,8,20],entir:[3,5,11,16],env:4,equal:5,equat:[0,1,2,7,10,12,13,14,15,20,21,22],equival:[6,11],error:5,essenti:4,evalu:[3,5,6,7,14,15],everi:[3,11],exampl:[0,2,3,4,5,6,7,8,10,11,12,13,14,15,16,18,19,20,21,23,24],excel:21,except:[0,3,16],execut:[0,1,3,4,7,8,10,11,14,20],exp:5,expand:6,expans:7,explain:1,explan:10,explicit:20,explicitli:[3,5,11],exploit:11,expos:[0,3,7,12,13,14,15,20],expr:[0,3,5,6,11,13,14,15],express:[0,3,5,6,7,11,13,14,15,20],exprs0:[14,15],exprs1:[14,15],extent:[3,8,11],exterior:16,extern:[12,13],extra:[7,20],extrem:[3,16],face:4,facilit:0,fact:7,factor:3,fall:[14,15],fals:[2,3,6,8,13,15,18],far:11,fd_order:6,featur:21,felt:4,few:4,fewer:[7,20],field:15,filter:0,find:10,finish:16,finit:[3,5,7,10,20,21,22],finite_differ:[6,7,18],first:[0,4,6,7,20],first_deriv:6,fit:3,float32:[2,7,8,12,13,14,15,20],focus:21,folder:4,follow:[0,3,4,5,11,16,21],forc:3,form:6,format:[7,12,13,14,15,20],forward:[4,6,11,19,20,24],found:21,four:11,free:[0,3,5],free_symbol:3,freeli:1,friction:4,from:[0,2,3,4,5,6,7,8,10,11,12,13,14,15,16,18,19,20,21,23,24],full:[0,21],funcptr:11,further:[7,20],fwi:21,gather:14,gaussian:0,gaussian_smooth:0,gener:[3,7,10,11,13,14,15,16,20],generic_deriv:6,geometri:8,get:[4,7,12,13,14,15,20,21],git:4,github:[4,21],given:[0,2,3,5,6,7,11,12,13,14,15,20],global:[7,8,11,20],gradual:21,greater:0,grid:[0,3,5,6,7,11,12,13,14,15,16,20,21,22],gridpoint:[12,13,14,15],group:21,guarante:[3,8],guard:[3,14,15],h_x:[6,7,8],h_y:[6,8],h_z:8,halo:[7,8,14,15,20],hand:[0,5],handi:4,handl:11,happi:4,has:[1,3,20],have:[0,4,10,18],help:[4,21],henc:3,here:[1,10,11,14,15,21],hierarchi:[3,7,12,13,14,15,20],high:10,highlight:21,hoc:11,honor:5,honour:3,how:[1,7,10,11,14,20,21],howev:[1,4,7,8],http:[0,4],i_m:3,ident:[7,20],illustr:4,impact:[6,7,20],imper:5,implement:[3,10,11,12,13,14,16,21],impli:11,implicit_dim:5,impos:11,inc:[5,11,13,15],includ:[0,4,7,10,11,20],incrdimens:3,increment:[3,5,11,13,15],index:[3,11,13,15],indic:[3,5,7,12,13,14,15,20],indirect:3,indirectli:3,individu:21,induc:11,info:[0,11],inform:[5,7,8,11,12,13,14,15,16,20],inhalo:[7,20],inherit:[8,18],initi:[0,7,12,13,14,15,20],initialis:0,initialize_funct:0,inject:[12,13,14,15],inlin:10,inner:[0,7,16,20],inner_i:16,inneri:16,innermost:0,input:[0,11,13,14,15,16],insert:[7,20],insid:4,inspect:[14,15],instal:10,instanti:3,instead:[3,7,12,13,14,15,20],instruct:[4,10],int32:0,integ:[7,11,20],integr:[3,5],interest:[7,20],interfac:[7,12,13,14,15,20],interior:[0,8,11,16],intermedi:20,interpol:[11,12,13,14,15],interpolation_coeff:[12,13],interpret:[2,7,8,12,13,14,15,20],interv:11,introduc:[14,15,21],introduct:21,invalid:5,involv:3,ipython:21,is_const:2,is_distribut:8,isn:[7,20],issu:4,iter:[3,11],its:[2,3,11,14],jit:[10,11],jupyt:[4,21],just:[3,10],keep:3,kei:[0,11,21],kernel:[0,10,11],keyword:[2,7,12,13,14,15,20],know:[10,11],known:[3,4],kwarg:[0,2,3,5,6,7,11,12,13,14,15,18,19,20,23,24],land:[2,14],languag:5,larger:4,largest:11,latter:6,layer:0,lazili:[7,12,13,14,15,20],lead:20,learn:10,least:4,left:[0,3,5,6,7,16,20],leftmost:3,len:0,level:[3,10,11],leverag:11,lhs:[0,5],like:[0,3,4,5,6,11,13,14,15,16],limit:[5,11],line:4,linear:[12,13,14,15,21],link:4,list:[0,4,5,11,21],live:14,load:4,local:[3,7,8,12,13,14,15,20],localhost:4,locat:[6,14],logic:[8,14],look:[3,4,7,10,12,13,14,15,20],loop:[3,11],lorena:21,lower:[1,3],lowest:[12,13],mai:[4,5,7,11,20],main:4,major:[7,12,13,14,15,20],make:3,mani:[0,3,11],map:[6,8],mapper:[0,16],march:11,match:[0,5],materi:10,matric:[18,19,23,24],matrix:[11,18,19,23,24],matvec:6,max:3,maxim:[7,20],maximum:[0,3,11],mean:4,memori:[7,11,12,13,14,15,20],memoryalloc:[7,12,13,14,15,20],met:[14,15],method:[3,4,11,16],middl:[3,16],might:4,migrat:21,min:3,miniconda:4,minimis:21,minimum:[0,3,11],mix:21,mmax:0,mmin:0,mode:[0,6],modifi:[2,7,12,13,14,15,20],modulo:3,modulodimens:3,more:[0,3,5,7,8,10,11,12,13,14,15,16,20],most:3,move:0,moving_averag:0,mpi:[0,7,8,11,12,13,14,15,20],multi:[7,12,13,14,15,20],multipl:[10,11],multipli:[12,13],must:[2,3,7,11,12,13,14,15,20],mutabl:3,mysiz:20,name:[0,2,3,5,6,7,8,11,12,13,14,15,16,20],navier:21,nbl:0,ndarrai:[0,7,11,12,13,14,15,20],ndim:[12,13],nearbi:[7,8,20],necessari:[4,7,12,13,14,15,20],need:[3,4,7,12,13,14,15,16,18,20],neumann:21,node:5,non:3,none:[0,3,6,7,8,13,14,15,16,20],nonlinear:21,norm:0,notat:[3,11],note:[0,2,3,4,5,6,7,8,12,13,14,15,16,20],notebook:[4,21],now:[0,7,11,16,20],npoint:[11,12,13,14,15],numa:[7,12,13,14,15,20],number:[0,3,5,7,8,11,12,13,14,15,20],numpi:[0,2,7,8,11,12,13,14,15,20],object:[0,2,4,5,7,8,11,12,13,14,15,16,20,22],obtain:6,occur:[5,7,12,13,14,15,20],occurr:5,offset:[3,8,13,14,15],often:8,onc:4,one:[0,3,7,11,12,13,14,15,16,20],onli:[3,4,5,6,7,8,11,12,13,14,15,20],onto:[12,13,14],open:4,openmp:11,oper:[2,3,5,7,12,13,14,15,20,21,22],operand:[0,5],oppos:11,opt:11,optim:[5,11],optimis:[10,11],option:[0,2,3,5,6,7,8,11,12,13,14,15,18,20],order:[0,4,5,6,7,11,12,13,14,15,20],org:0,origin:[6,8,12,13],origin_dim:6,origin_offset:8,other:[3,4,5,7,20],otherwis:[2,3,8,11],our:[4,10,21],out:[3,4,11,21],outer:0,outhalo:[7,12,13,14,15,20],output:11,over:[0,3,8,11],overrid:[3,16],overridden:16,own:[3,4,12,13,14,15],ownership:14,p_sf:[14,15],p_t:[13,15],packag:4,pad:[0,7,20],padfunc:0,paper:1,parallel:11,paramet:[0,2,3,5,6,7,8,11,12,13,14,15,18,20],parametr:16,parent:3,pars:5,part:4,particular:5,particularli:14,pass:[0,3,4,6,7,11,20],password:4,past:4,per:[7,8,12,13,20],perform:[0,3,11,13,15],pertain:0,physic:[3,7,8,11,12,13,14,15,20],place:11,platform:[10,11],pleas:4,plu:0,point:[3,4,7,8,11,12,13,14,15,16,20],poli:3,polytool:3,port:4,posit:[7,20],possibl:4,potenti:[12,13],precomputedsparsefunct:[13,17,22],precomputedsparsetimefunct:[17,22],present:[4,11,20],preset:16,press:4,problem:[3,8,10,21],proce:[4,11],process:8,produc:[3,6,16],product:[0,4],program:5,progress:1,project:4,prompt:4,properti:[2,3,5,7,8,11,12,13,14,15,18,19,20,23,24],protocol:20,provid:[0,2,3,4,7,8,11,12,13,14,15,20],pseudocod:3,pure:21,python:[2,4,21],quadrant:11,quick:21,rais:[0,5],rang:3,rank:[7,14,20,23],rare:3,rather:[3,13,15,21],read:[7,12,13,14,15,20],rearrang:5,recommend:4,reduct:11,refer:[3,5,7,8,10,11,12,13,14,15,16,20],reflect:0,region:[0,3,5,7,8,11,12,13,14,15,16,20],relat:[4,5],relax:[14,15],relev:[8,11],remain:6,replac:[5,11],repositori:21,repres:[2,3,7,8,12,13,14,15,18,20],represent:7,requir:[0,4,5,20],respect:[3,11],restrict:5,result:[5,6,13,15,20],retriev:0,revers:21,review:1,rhs:[0,5],rhss:0,right:[0,3,5,6,7,14,16,20],root:3,rout:4,routin:[0,14,15],row:[7,12,13,14,15,18,19,20,23,24],rule:5,run:[0,4,11,14],runtim:11,same:[11,18],sampl:[4,12,13,14,15],save:[3,11,20],scalar:[0,2,11],scatter:14,scenario:[7,20],scheme:6,scientif:21,second:[0,6,7],second_deriv:6,see:[3,4],seen:11,self:[3,7,12,13,14,15,16,20],semant:[6,11],sensit:4,sequenc:[3,11],seri:[1,21],server:4,session:4,set:[0,3,4,8,11,14,16,21],setup:4,sever:21,sf_coord:[14,15],shape:[0,3,5,6,7,8,11,12,13,14,15,18,19,20,23,24],shape_alloc:[7,20],shape_glob:[7,20],shape_loc:8,shape_with_halo:[7,8,20],shell:4,shift:6,shifteddimens:3,shortcut:[3,6],should:[3,5,7,11,16,20],shouldn:[7,20],show:11,shown:3,side:[0,5,6,7,20],sigma:0,simd:11,simpl:[0,11,21],simple_moving_averag:0,simpli:[4,20],sin:5,sinc:[2,4,5,7,12,13,14,15,20],singl:[3,8,12,13],size:[0,3,7,8,18,20],slack:4,smallest:11,smooth:0,softwar:10,solv:[5,21],some:[3,4,14,15,21],sometim:[7,20],sourc:[0,2,3,4,5,6,7,8,11,12,13,14,15,16,18,19,20,23,24],space:[3,7,8,12,13,14,15,20,23],space_dimens:[7,20],space_ord:[6,7,12,13,14,15,20],spacedimens:[3,8],spacing_map:8,spacing_symbol:8,span:[3,16],spars:[11,12,13,14,15],sparsefunct:[0,5,11,12,13,15,17,22],sparsetimefunct:[13,17,22],spatial:[3,7,8,20],special:3,specifi:[0,5,8,11,14,15,20],split:[12,13],stagger:[6,7,18,20,21],standard:[0,14,15],start:[4,11,16,21],state:[3,19,20,24],stencil:[6,8,20],step:[3,8,11,21],stepping_dim:[3,8,11],steppingdimens:3,still:1,stoke:21,stop:4,store:[7,12,13,14,15,20],str:[0,2,3,7,11,12,13,14,15,20],structur:21,sub:[0,3,5,11],subclass:16,subdimens:[3,11],subdomain:[0,3,5,8,9,11,22],subexpress:5,submit:1,subsampl:3,substitut:[5,11],succeed:4,successfulli:4,suffic:[7,20],suitabl:11,sum:[0,7,20],sumal:0,summand:[7,20],summari:11,summat:5,suppli:11,support:[0,4,11,14,15],sure:3,surround:[7,8,20],symbol:[1,2,3,5,6,7,8,10,11,12,13,14,15,19,20,22,24],symbolic_incr:3,symbolic_max:3,symbolic_min:3,symbolic_s:3,symmetr:18,sympi:[2,3,5,7,10,12,13,14,15,18,19,20,23,24],syntax:20,system:[4,5],take:[7,12,13,14,15,16,18,20],target:5,tempor:20,temporari:14,tensor:[5,7,11,12,13,14,15,18,19,20,23,24],tensorfunct:[17,19,22,23],tensortimefunct:[17,22,24],termin:4,test:4,than:[0,3,13,15,21],thei:[3,4,21],themselv:5,theori:21,therefor:[7,20],thi:[0,1,3,4,5,6,7,8,10,11,12,13,14,15,16,20],thick:[3,16],thing:[3,4],those:[3,4,7,20],thought:5,three:[3,4,12,13,16],through:[0,7,11,20,21],thu:[2,3],time:[3,7,8,10,11,13,15,19,20,21,24],time_dim:[3,8,20],time_dimens:8,time_m:11,time_ord:[13,15,20],timedimens:[3,8,20],timefunct:[7,8,11,15,17,18,22],timestep:[11,15],token:4,toler:[14,15],tom:1,too:3,tool:5,topic:21,topolog:8,total:21,track:3,transpos:6,tree:5,tri:[14,15],trivial:11,truncat:0,tupl:[0,3,6,7,8,12,13,14,15,18,19,20,23,24],tutori:[1,4,10],two:[0,3,5,8,12,13],type:[2,3,5,6,7,8,12,13,14,15,16,18,19,20,23,24],typic:[3,4,7,11,20],u_t:[13,15],under:1,underli:8,unevalu:6,uniqu:16,unit:[3,4,8,11],unlik:[12,13],updat:[8,14],upon:[7,14,15],upper:3,url:4,use:[3,4,7,10,12,13,14,15,20,21],used:[0,3,5,7,8,11,12,13,14,15,20],user:[0,2,3,4,5,11,14],uses:[2,3,4,7,11,12,13,14,15,20],using:[4,11,12,13,14,15,20],valu:[0,2,3,7,8,11,12,13,14,15,18,20,23],valueerror:[0,5],vari:[7,13,15,19,20,23,24],variabl:[3,11],variat:21,vector:[11,23],vectorfunct:[17,22,24],vectoris:11,vectortimefunct:[17,19,22],veri:20,version:[4,7,20],via:[0,3,4,6],view:[7,12,13,14,15,20],volum:8,volume_cel:8,wai:4,want:[7,10,12,13,14,15,20],waveform:21,weight:5,welcom:[1,10],well:[3,7,20],were:4,what:4,when:[3,4,5,7,12,13,14,15,20],where:[4,11,14],whether:[4,18],which:[0,3,5,6,7,8,10,11,12,13,14,15,16,20],whistl:4,width:6,wiki:0,wikipedia:0,wildcard:5,window:4,wise:0,wish:0,with_halo:[7,20],within:[2,3,5,11,14,15],work:[1,10],would:[4,5,12,13],wouldn:11,wrapper:5,write:20,written:[7,12,13,14,15,20],x_m:11,xreplac:5,y_m:11,yml:4,you:[4,7,10,12,13,14,15,20,21],your:4,zero:[18,19,23,24]},titles:["Built-in Operators","The Devito Compiler","Constant","Dimension","Obtain Devito","Equation","Finite-difference approximations","Function","Grid","Grids","The Devito project","Operator","PrecomputedSparseFunction","PrecomputedSparseTimeFunction","SparseFunction","SparseTimeFunction","SubDomain","Symbolic Objects","TensorFunction","TensorTimeFunction","TimeFunction","Tutorials","API Reference","VectorFunction","VectorTimeFunction"],titleterms:{"function":7,The:1,api:22,approxim:6,built:0,compil:1,comput:21,conda:4,constant:2,devito:[1,4,10],differ:6,dimens:3,docker:4,dynam:21,environ:4,equat:5,finit:6,fluid:21,get:10,grid:[8,9],instal:4,invers:21,model:21,object:17,obtain:4,oper:[0,11],pip:4,precomputedsparsefunct:12,precomputedsparsetimefunct:13,refer:22,seismic:21,sparsefunct:14,sparsetimefunct:15,start:10,subdomain:16,symbol:17,tensorfunct:18,tensortimefunct:19,timefunct:20,tutori:21,vectorfunct:23,vectortimefunct:24}})