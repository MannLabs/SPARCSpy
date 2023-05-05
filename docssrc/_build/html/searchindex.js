Search.setIndex({docnames:["index","pages/guides/basic_workflow","pages/ml","pages/pipeline/base","pages/pipeline/extraction","pages/pipeline/introduction","pages/pipeline/project","pages/pipeline/protocols","pages/pipeline/segmentation","pages/pipeline/selection","pages/processing/preprocessing","pages/vipercmd"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["index.rst","pages/guides/basic_workflow.rst","pages/ml.rst","pages/pipeline/base.rst","pages/pipeline/extraction.rst","pages/pipeline/introduction.rst","pages/pipeline/project.rst","pages/pipeline/protocols.rst","pages/pipeline/segmentation.rst","pages/pipeline/selection.rst","pages/processing/preprocessing.rst","pages/vipercmd.rst"],objects:{"vipercore.ml":{datasets:[2,0,0,"-"],metrics:[2,0,0,"-"],models:[2,0,0,"-"]},"vipercore.ml.datasets":{HDF5SingleCellDataset:[2,1,1,""],NPYSingleCellDataset:[2,1,1,""]},"vipercore.ml.metrics":{auc:[2,2,1,""],precision:[2,2,1,""],precision_top_n:[2,2,1,""],recall:[2,2,1,""],recall_top_n:[2,2,1,""]},"vipercore.ml.models":{GolgiCAE:[2,1,1,""],GolgiVAE:[2,1,1,""],GolgiVGG:[2,1,1,""]},"vipercore.ml.models.GolgiCAE":{forward:[2,3,1,""]},"vipercore.ml.models.GolgiVAE":{decode:[2,3,1,""],encode:[2,3,1,""],forward:[2,3,1,""],loss_function:[2,3,1,""],reparameterize:[2,3,1,""]},"vipercore.ml.models.GolgiVGG":{forward:[2,3,1,""]},"vipercore.pipeline":{project:[6,0,0,"-"]},"vipercore.pipeline.base":{Logable:[7,1,1,""],ProcessingStep:[7,1,1,""]},"vipercore.pipeline.base.Logable":{DEFAULT_FORMAT:[7,4,1,""],DEFAULT_LOG_NAME:[7,4,1,""],directory:[7,4,1,""],log:[7,3,1,""]},"vipercore.pipeline.base.ProcessingStep":{log:[7,3,1,""]},"vipercore.pipeline.project":{Project:[6,1,1,""]},"vipercore.pipeline.project.Project":{DEFAULT_CLASSIFICATION_DIR_NAME:[6,4,1,""],DEFAULT_CONFIG_NAME:[6,4,1,""],DEFAULT_EXTRACTION_DIR_NAME:[6,4,1,""],DEFAULT_SEGMENTATION_DIR_NAME:[6,4,1,""],DEFAULT_SELECTION_DIR_NAME:[6,4,1,""],classify:[6,3,1,""],extract:[6,3,1,""],load_input_from_array:[6,3,1,""],load_input_from_file:[6,3,1,""],segment:[6,3,1,""],select:[6,3,1,""]},"vipercore.pipeline.segmentation":{Segmentation:[8,1,1,""],ShardedSegmentation:[8,1,1,""]},"vipercore.pipeline.segmentation.Segmentation":{DEFAULT_FILTER_FILE:[8,4,1,""],DEFAULT_OUTPUT_FILE:[8,4,1,""],PRINT_MAPS_ON_DEBUG:[8,4,1,""],call_as_shard:[8,3,1,""],identifier:[8,4,1,""],initialize_as_shard:[8,3,1,""],input_path:[8,4,1,""],load_maps_from_disk:[8,3,1,""],log:[8,3,1,""],save_map:[8,3,1,""],save_segmentation:[8,3,1,""],window:[8,4,1,""]},"vipercore.pipeline.segmentation.ShardedSegmentation":{DEFAULT_FILTER_FILE:[8,4,1,""],DEFAULT_INPUT_IMAGE_NAME:[8,4,1,""],DEFAULT_OUTPUT_FILE:[8,4,1,""],DEFAULT_SHARD_FOLDER:[8,4,1,""],log:[8,3,1,""],resolve_sharding:[8,3,1,""]},"vipercore.pipeline.selection":{LMDSelection:[9,1,1,""]},"vipercore.pipeline.selection.LMDSelection":{check_cell_set_sanity:[9,3,1,""],load_classes:[9,3,1,""],log:[9,3,1,""],process:[9,3,1,""]},"vipercore.processing":{preprocessing:[10,0,0,"-"]},"vipercore.processing.preprocessing":{percentile_normalization:[10,2,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","function","Python function"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:function","3":"py:method","4":"py:attribute"},terms:{"0":[2,5,6,8,9,10,11],"000":5,"00028":[],"001":5,"01":2,"02":[],"03":5,"05":[5,11],"1":[2,5,6,8,9,10,11],"10":[5,9],"100":[5,9],"1000":[5,6,8,11],"100000000":5,"1024":5,"1100":9,"12":[],"128":5,"15":5,"150":5,"1500":5,"16":[],"1704":[],"196":2,"1e":[],"2":[2,5,6,9],"20":9,"200":5,"2000":[6,8],"2021_09_03_hdf5_extraction_develop":5,"2021_11_13_pipeline_test":[],"2250000":5,"25":9,"277":11,"3":[3,4,5,7,9],"30":[5,9,11],"300":9,"3010":5,"358":11,"3gib":11,"4":[5,9,11],"4000":5,"41":5,"468":11,"4gib":11,"5":[2,5,9,11],"50":5,"5000":5,"50000":5,"505":9,"5242880000":5,"5gb":5,"5gib":11,"6":9,"69":11,"6gib":11,"7":[5,9],"72":11,"729":11,"731":11,"753":11,"755":11,"775":11,"8":[5,11],"80":5,"81":[],"9":[5,11],"92":5,"95":5,"97":[],"98":5,"99":[],"999":5,"case":[6,8],"class":[2,3,4,5,6,7,8,9],"default":[3,4,6,7,8,11],"do":8,"final":5,"float":[8,10],"function":[2,6,8,9],"import":5,"int":[2,5,6,8],"new":[6,8],"return":[2,8,9],"true":[3,4,5,6,7,11],"var":2,"while":2,A:[3,4,7,9],At:[],But:[],For:[5,6,8,9,11],If:[5,6,9,11],In:8,Is:[3,4,7],It:8,NOT:6,OR:[],The:[3,4,5,6,7,8,9,10],There:[],These:8,To:[6,8],_:[3,4,7],a1:[5,9],ab:[],about:6,abov:5,absolut:11,acc:[],access:9,accuraci:[],across:[],actual:[],adam:[],adapt:5,add:[],add_imag:[],addit:[],aemodel:[],after:[],afterward:2,algorithm:[],all:[2,8,9,10,11],allow:5,along:[],alreadi:8,also:[],altern:9,although:2,amount:5,an:[5,8,9,10],ani:2,anyth:[],append:[],appli:8,applic:[3,4,6,7],approxim:9,ar:[3,4,5,7,8,9],area:[2,5,9],arg1:2,arg2:2,arg:[2,6,8,9,10],argmax:[],argument:9,arrai:[5,6,8,9,10],arxiv:[],assign:8,associ:[],attribut:8,auc:2,autoencodermodel:[],automat:[],automaticli:9,avail:8,averag:[5,9],b:2,back:[],background:5,backprop:[],backward:[],bar:[],base:[0,6,8,9,10],batch:[],batch_idx:[],been:8,befor:[3,4,5,7],begin:[],being:[],below:[],between:[5,9,10],binari:9,binary_smooth:9,bit:[],block:[],bool:[2,3,4,6,7,8],c:[2,11],cach:5,calcul:[5,10],calibr:[5,9],calibration_mark:[5,9],call:[2,3,4,6,7,8],call_as_shard:8,can:[3,4,5,6,7,8,9,11],care:2,cell:[6,8,9,11],cell_set:9,cells_to_select:[5,9],cfg:2,chang:[5,6],channel:[5,6,8,9,10],channel_remap:5,characterist:2,check:[8,9],check_cell_set_san:9,checkpoint:8,choos:5,choosen:5,chunk:5,chunk_siz:5,class_subset:9,classif:6,classifi:6,classification_f:6,clip:10,closer:9,closur:[],code:2,collaps:9,collect:11,column:[5,9],command:[],complet:[3,4,6,7],compress:[5,9,11],comput:[2,5,8],config:[3,4,6,7,9],config_path:[5,6],configur:6,configure_optim:[],confluent:5,consid:5,consol:[3,4,7],contact:5,contact_filt:5,contain:[6,8,9,11],continu:[],control:[],convolution_smooth:9,coordin:9,copi:6,core:[],corner:[6,8],correct:5,correspond:[5,8,9],cosineann:[],could:8,crash:[],creat:[3,4,6,7,8],creation:8,crop:[6,8],csv:[5,8,9],csv_locat:5,cuda:[],current:8,current_step:8,curv:[2,9],custom:8,cut:[5,9],cycl:[],cytosol:[5,9],d:2,dapi:5,data:[5,6,8,9],dataload:[],dataloader_idx:[],datapoint:[5,9],dataset1:[5,9],dataset2:9,dataset3:9,dataset:[5,11],date:[3,4,7,8],datetim:[3,4,7],debug:[3,4,6,7],decid:[],decim:9,declar:8,decod:2,decompress:11,def:[8,9],default_classification_dir_nam:6,default_config_nam:6,default_extraction_dir_nam:6,default_filter_fil:8,default_format:[3,4,7],default_input_image_nam:8,default_log_nam:[3,4,7],default_output_fil:8,default_segmentation_dir_nam:6,default_selection_dir_nam:6,default_shard_fold:8,defin:[2,5,6,8,9,10],definit:[9,11],delet:[3,4,6,7],delt:[],depend:9,deriv:8,descend:[3,4,7],describ:[],descript:2,desir:9,detect:[],deviat:2,dict:[3,4,7,8,9],dictionari:9,differ:6,dilat:[5,9],dim:[],dimens:[2,6,8],dir_label:[],dir_list:[],directori:[3,4,6,7,8,9,11],dis_opt:[],dis_sch:[],disabl:[],disk:[5,8],displai:[],distanc:[5,9],distance_heurist:9,doc:[3,4,7],document:5,doe:[3,4,6,7],don:[],download_testimag:5,downsampl:5,dss:[],dss_fast:5,dure:8,e:8,each:6,edg:9,either:[],el:5,element:[6,8],els:11,enabl:[],encod:2,end:8,entir:9,entri:[3,4,7,8],epoch:[],equal:5,eros:[5,9],especi:6,establish:9,eval:[],ever:5,everi:[2,3,4,7],exampl:[5,6,8,9,11],example_imag:[],exhaust:9,exist:[3,4,6,7,11],expect:6,experi:[],exponentiallr:[],extend:2,extract:[5,6,8],extraction_f:[5,6],f:5,factor:9,fals:[3,4,6,7,8,11],fancier:[],feder:8,field:[],file:[3,4,6,7,8,9],file_path:6,filter:[5,8],find:6,first:[5,6,8,9],fit:[],five_class:[],flag:6,focu:9,fold:[5,9],folder:[6,11],foldernam:6,follow:[5,9],form:8,format:[3,4,7,8,9],former:2,forward:[2,9],found:8,fraction:11,frequenc:[],friedn:[],from:[2,3,4,5,6,7,8,9],further:8,g:8,gan:[],gaussian:2,gen_opt:[],gen_sch:[],gener:[5,8,9],generalmodel:[],georgwallmann:5,get:8,get_output:9,given:2,global:[5,9],goe:[],golgi_lib_nig:[],golgica:2,golgiva:2,golgivgg:2,gpu:[],gradient:[],greedi:9,greedili:9,greedy_k:9,grid:[],h5:[8,11],h:[2,11],ha:8,halv:5,handl:[],have:8,hdf5:[5,8,9,11],hdf5_rdcc_nbyte:5,hdf5_rdcc_nslot:5,hdf5_rdcc_w0:5,hdf5cellextract:5,hdf5singlecelldataset:2,hdf_locat:9,height:[6,8,10],hello:[],help:5,helper:8,here:[],heurist:9,hidden:[],hidden_dim:2,high:5,higher:10,hilbert:9,hilbert_p:9,hood:[],hook:2,how:[],hparam:[],html:[3,4,7],http:[3,4,7],id:8,identifi:[8,9],ignor:2,im:10,imag:[2,5,6,8,10],image_s:5,img:6,implement:9,improv:9,in_channel:2,inbetween:9,includ:[],increas:5,index:8,indic:8,individu:9,inference_devic:[],inform:11,initi:8,initialize_as_shard:8,input:[2,6,8,10,11],input_channel:[5,6],input_dataset:11,input_imag:8,input_path:8,input_segment:9,instanc:2,instead:[2,9],integ:[8,9],intent:8,interest:[],intermedi:[3,4,6,7,8],intermediate_output:[3,4,6,7],intern:8,interpret:11,interv:[],item:[],iter:8,its:[3,4,7,8,11],join:5,k:9,kei:9,keyword:[],kl:2,know:[],known:9,kwarg:[2,6,8,9],label:[2,5,8,9],labels_hat:[],larger:5,last:[],latent:2,latent_dim:2,latter:2,lbfg:[],lead:9,learn:[],learningratemonitor:[],least:9,left:[5,6,8],leica:9,len:[],length:[6,11],level:11,librari:[3,4,7],lightn:[],lightningmodul:[],like:6,line:2,list:[2,3,4,5,6,7,8,9],live:[3,4,7],lmd:9,lmd_object:[],lmdselect:[0,5],load:[3,4,6,7,8,9],load_class:9,load_input_from_arrai:6,load_input_from_fil:[5,6],load_maps_from_disk:8,local:5,locat:[5,8],location_path:6,log:[2,3,4,7,8,9],log_dict:[],logabl:0,logger:[],logvar:2,loop:[],loss:2,loss_funct:2,lower:[5,10],lower_percentil:10,lower_quantile_norm:5,lr:[],lr_dict:[],lr_schedul:[],lstm:[],lzf:11,make:8,make_grid:[],manipul:11,manual:6,map0:8,map1:8,map:[2,8],map_nam:8,marker:[5,9],marker_0:[5,9],marker_1:[5,9],marker_2:[5,9],mask:[5,9],matrix:9,max_clip:5,max_level:[],max_siz:5,maximum:5,mean:2,median:5,median_block:5,median_filter_s:5,median_step:5,member:[],membrane_multilabel_cnn:[],memori:5,mention:[],merg:9,messag:[3,4,7,8,9],method:[],metric:[],microscop:9,might:6,min_clip:5,min_dist:5,min_siz:5,minimum:5,miss:5,mitig:9,ml:[],mm:9,mnt:5,mode:[],model:[],model_di:[],model_gen:[],modul:2,monitor:[],more:6,most:[8,9],mostli:8,mu:2,multi:[],multilabelsupervisedmodel:[],multipl:11,multithread:5,must:[3,4,7],my:[],mymap:8,mynumpyarrai:8,n:[2,5,9],n_critic:[],name:[3,4,5,6,7,8,9],ndarrai:6,nearest:9,need:[2,5,6,8,9],neighbour:9,network:2,newli:[3,4,6,7,8],next:[],nn:[],none:[2,6,8,9],normal:[5,10],note:5,np:[5,8,9,10],npysinglecelldataset:2,nth:5,nuclear:[5,9],nuclei:5,nucleu:5,nucleus_segment:5,num_class:2,number:[5,6,9,11],numpi:[5,6,8,9,10],o:11,obj:[],object:[3,4,7,8],often:[],on_train_epoch_end:[],on_train_epoch_start:[],on_train_start:[],on_validation_epoch_end:[],on_validation_epoch_start:[],one:2,onli:[5,8,9],onto:2,open:5,oper:2,optim:9,optimizer_idx:[],optimizer_on:[],optimizer_step:[],optimizer_two:[],option:[6,9],order:[5,6,9],org:[3,4,7],os:5,otsu:5,out:6,out_channel:2,output:[2,3,4,6,7,8,11],over:[5,8,9],overlap:9,overrid:6,overridden:2,overwrit:[3,4,5,6,7],own:[3,4,7],p:9,page:[],parallel:8,param:[2,3,4,7],paramet:[2,3,4,5,6,7,8,9],paramref:[],paramt:9,part:9,pass:[2,3,4,7,8,9],path1:6,path2:6,path3:6,path:[5,6,9],path_optim:9,pca:[],pca_dimens:[],peak_footprint:5,per:5,percentag:2,percentil:[5,10],percentile_norm:10,perform:[2,5,8],pipelin:[3,4,5,6,7,8,9],pixel:[5,9],place:9,plan:8,pleas:5,plmodel:[],point:[6,8,9],poly_compression_factor:[5,9],pos_label:2,posit:[],possibl:[6,8],precis:2,precision_top_n:2,predict:2,predictiong:2,preprocess:0,present:[],previou:9,print:[3,4,5,6,7],print_maps_on_debug:8,problem:9,procedur:[],process:[3,4,5,6,7,8,9,10],processingstep:[0,8,9],produc:[],progress:11,project:[0,3,4,5,7,9,11],project_loc:5,propag:[],provid:9,pseudocod:[],put:[],px:[6,8,9],python:[3,4,7],pytorch:[],pytorch_lightn:[],quantil:5,r:11,rac:2,randint:[],random:11,randomli:11,rate:[],read:[5,8,9],reader:5,recal:2,recall_top_n:2,receiv:2,recent:9,recip:2,recommend:[5,9],recurs:11,recursivli:9,reducelronplateau:[],reduct:[5,9],refer:[6,8],regist:2,rel:9,relativ:9,relev:8,remap:[5,6],remov:[5,9],renam:6,reparameter:2,repeat:9,requir:9,resolut:9,resolve_shard:8,rest:[],result:[8,11],resum:8,retriv:[],return_fake_id:[],return_id:[],right:9,root_dir:[],round:9,row:[5,9],run:2,s:[3,4,7,8,9],same:5,sampl:2,sample_img:[],saniti:[],save:[3,4,6,7,8],save_map:8,save_segment:8,scale:9,scan:11,schedul:[],screen_label:[],search:[],search_directori:11,second:[5,6,9],section:6,see:[3,4,6,7],segment:[0,5,6,9],segmentation_channel:[5,9],segmentation_f:[5,6,9],select:[0,5,6],select_channel:[],selection_f:[5,6,9],self:[8,9],sequenti:[],set:[3,4,5,6,7,8,9],sgd:[],shape:[6,8,9,10],shape_dil:[5,9],shape_gener:[],shard:[5,8],shard_siz:5,shardedsegment:0,shardedwgasegment:5,sharding_plan:8,should:[2,3,4,5,6,7,9,11],show:[0,11],shown:[],shrink:5,shuffl:11,sigma:2,silent:2,similar:[5,9],sinc:2,singl:[6,9,11],single_cel:11,size:[5,9],skip:[],slide000:11,slide001:11,slide:9,smooth:[5,9],smoothing_filter_s:5,so:[6,8],solv:9,some:[8,11],someth:[],sort:9,space:2,specif:5,specifi:6,speckel:5,speckl:5,speckle_kernel:5,speed:5,split:0,spread:9,standard:2,start:[8,9],stat:0,state:8,step:[3,4,5,6,7,8],stitch:8,storag:5,str:[2,3,4,6,7,8,9],strftime:[3,4,7],strict:[],string:[3,4,7,8,9],strong:5,stuff:8,subclass:2,subdirectori:[3,4,7],sum:11,summari:2,suppli:[6,9],support:[],system:9,t:11,t_max:[],take:[2,11],target:2,tell:[],temp:5,tensor:2,test:11,testdaten:5,text:[],than:9,thei:8,them:2,therefor:9,thi:[2,6,8,9,11],thing:[],third:9,those:[],thread:[5,9,11],threshold:5,through:2,tiff:6,tile:[5,6,8],time:[3,4,7,8,9,11],togeth:9,tool:[],top:[6,8],top_n:2,torch:[],torchvis:[],total:9,tpu:[],train:11,training_epoch_end:[],training_step:[],transform:9,tremend:5,treshold:5,tri:8,trick:2,truncat:[],truncated_bptt_step:[],tsne:[],tune:9,tupl:[6,8],two:[5,8,9],type:[2,3,4,6,7,8,9],um:9,umap:[],umap_min_dist:[],umap_neighbour:[],under:[2,6],uneven:5,uniqu:8,unit:[],until:9,up:5,updat:[],upper:[5,10],upper_percentil:10,upper_quantile_norm:5,us:[3,4,5,6,7,8,9,10,11],usag:11,user:5,util:5,vae:2,val:[],val_acc:[],val_batch:[],val_data:[],val_loss:[],val_out:[],valid:[6,9,11],validation_epoch_end:[],validation_step:[],validation_step_end:[],valu:[2,8,10,11],veri:[],version_0:[],viper:6,viper_library_test:5,vipercor:[2,3,4,5,6,7,8,9,10],w:2,wasserstein:[],we:9,well:[5,9],were:8,wga:[5,9],wga_segment:5,what:[],whatev:[],when:[3,4,5,6,7,9],where:[3,4,6,7,8],whether:[],which:[3,4,5,6,7,8,9,11],whoch:[],whole:9,whose:[],wide:6,width:[6,8,10],window:8,wise:10,within:2,without:[],workflow:[5,6,8,9],wrapper:8,written:8,x:[2,9],xml_decimal_transform:9,y:[],yaml:5,yet:[3,4,7],yml:6,you:[],your:[],z:2,zero:8},titles:["Welcome to Viper Core\u2019s documentation!","&lt;no title&gt;","ml","base","base","base","project","base","segmentation","selection","<span class=\"section-number\">1. </span>Preprocessing","viper-split"],titleterms:{"2":[],argument:11,base:[3,4,5,7],command:0,content:0,core:0,dataset:2,document:0,indic:[],line:0,lmd:[],lmd_object:[],lmdselect:9,logabl:[3,4,7],metric:2,ml:2,model:2,modul:[],name:11,pipelin:0,plmodel:2,posit:11,preprocess:10,process:0,processingstep:[3,4,7],project:6,s:0,segment:8,select:9,shardedsegment:8,split:11,stat:11,tabl:0,tool:0,viper:[0,11],welcom:0}})