unit specialcommands;
{
  Special commands are commands on the GPU stack that due to technical
  reasons cannot be executed in plugins. They need to be executed
  by the GPU core
  
  (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM
  This unit is released under GNU Public License (GPL)
}
interface

uses SysUtils,
     stacks, identities, stkconstants, pluginmanagers, frontendmanagers,
     methodcontrollers, resultcollectors, plugins;

type TSpecialCommand = class(TObject)
 public
   constructor Create(var plugman : TPluginManager; var meth : TMethodController;
                      var res : TResultCollector; var frontman : TFrontendManager);
   destructor  Destroy();

   function isSpecialCommand(arg : String; var specialType : TStkArgType) : boolean;
   
   function execUserCommand(arg : String; var stk : TStack) : boolean;
   function execNodeCommand(arg : String; var stk : TStack) : boolean;
   function execThreadCommand(arg : String; var stk : TStack) : boolean;
   function execCoreCommand(arg : String; var stk : TStack) : boolean;
   function execPluginCommand(arg : String; var stk : TStack) : boolean;
   function execFrontendCommand(arg : String; var stk : TStack) : boolean;
   function execResultCommand(arg : String; var stk : TStack) : boolean;


 private
   plugman_      : TPluginManager;
   meth_         : TMethodController;   
   rescollector_ : TResultCollector;
   frontman_     : TFrontendManager;
end;

implementation

constructor TSpecialCommand.Create(var plugman : TPluginManager; var meth : TMethodController;
                                   var res : TResultCollector; var frontman : TFrontendManager);
begin
 inherited Create();
  plugman_      := plugman;
  meth_         := meth;
  rescollector_ := res;
  frontman_     := frontman;
end;

destructor TSpecialCommand.Destroy();
begin
  inherited;
end;

function TSpecialCommand.isSpecialCommand(arg : String; var specialType : TStkArgType) : boolean;
begin
  Result := false;
  specialType := STK_ARG_UNKNOWN;
  if (arg='user.id') or (arg='user.name') or (arg='user.email') or (arg='user.homepage_url') or
     (arg='user.realname')  then
      begin
	    specialType := STK_ARG_SPECIAL_CALL_USER;
	    Result := true;
	  end
  else
  if (arg='node.name') or (arg='node.team') or (arg='node.country') or (arg='node.region')  or
     (arg='node.id') or (arg='node.ip') or (arg='node.os') or (arg='node.version')  or
	 (arg='node.accept') or (arg='node.mhz') or (arg='node.ram') or (arg='node.gflops')  or
	 (arg='node.issmp') or (arg='node.isht') or (arg='node.is64bit') or (arg='node.iswine')  or
	 (arg='node.isscreensaver') or (arg='node.cpus') or (arg='node.uptime') or (arg='node.totuptime') or 
	 (arg='node.cputype') or (arg='node.localip') or (arg='node.longitude') or (arg='node.latitude') or
     (arg='node.port') then
      begin
	    specialType := STK_ARG_SPECIAL_CALL_NODE;
	    Result := true;
	  end
  else
  if (arg='thread.sleep')  then
      begin
        specialType := STK_ARG_SPECIAL_CALL_THREAD;
	    Result := true;
      end
  else	  
  if (arg='core.threads') or (arg='core.maxthreads') or (arg='core.isidle') or (arg='core.hasresources') or
     (arg='core.version') or (arg='core.registeredjobs') or (arg='core.stkversion') or
     (arg='core.downloads') or (arg='core.maxdownloads') or (arg='core.services') or (arg='core.maxservices')
     then
      begin
            specialType := STK_ARG_SPECIAL_CALL_CORE;
	    Result := true;
      end
  else	  
  if (arg='plugin.load') or (arg='plugin.discard')  or (arg='plugin.getfield') or
     (arg='plugin.list') or (arg='plugin.isloaded') or (arg='plugin.which') or (arg='plugin.isable') then
      begin
        specialType := STK_ARG_SPECIAL_CALL_PLUGIN;
	    Result := true;      
      end
  else
  if (arg='frontend.udp.register') or (arg='frontend.files.register') or 
     (arg='frontend.unregister') or (arg='frontend.list') then 
     begin
        specialType := STK_ARG_SPECIAL_CALL_FRONTEND;
	    Result := true;      
     end
  else
  if (arg='result.last') or (arg='result.avg') or (arg='result.n') or (arg='result.nfloat') or
     (arg='result.sum') or (arg='result.min') or (arg='result.max') or (arg='result.first') or
     (arg='result.stddev') or (arg='result.variance') or (arg='result.history') or
	 (arg='result.overrun') then
     begin
        specialType := STK_ARG_SPECIAL_CALL_RESULT;
	Result := true;
     end;     
  
end;


function TSpecialCommand.execUserCommand(arg : String; var stk : TStack) : boolean;
begin
  Result := false;
  if (arg='user.id') then Result := pushStr(MyUserID.userid, stk) else
  if (arg='user.name') then Result := pushStr(MyUserID.username, stk) else
  if (arg='user.email') then Result := pushStr(MyUserID.email, stk) else
  if (arg='user.realname') then Result := pushStr(MyUserID.realname, stk) else
  if (arg='user.homepage_url') then Result := pushStr(MyUserID.homepage_url, stk) else
    raise Exception.Create('User argument '+QUOTE+arg+QUOTE+' not registered in specialcommands.pas');
  Result := true;  
end;

function TSpecialCommand.execNodeCommand(arg : String; var stk : TStack) : boolean;
begin
  Result := false;
  if (arg='node.name')          then Result := pushStr(MyGPUID.nodename, stk) else
  if (arg='node.team')          then Result := pushStr(MyGPUID.team, stk) else
  if (arg='node.country')       then Result := pushStr(MyGPUID.country, stk) else
  if (arg='node.region')        then Result := pushStr(MyGPUID.region, stk) else
  if (arg='node.id')            then Result := pushStr(MyGPUID.nodeid, stk) else
  if (arg='node.ip')            then Result := pushStr(MyGPUID.ip, stk) else
  if (arg='node.port')          then Result := pushFloat(MyGPUID.port, stk) else
  if (arg='node.os')            then Result := pushStr(MyGPUID.os, stk) else
  if (arg='node.version')       then Result := pushStr(MyGPUID.version, stk) else
  if (arg='node.accept')        then Result := pushBool(MyGPUID.acceptincoming, stk) else
  if (arg='node.mhz')           then Result := pushFloat(MyGPUID.mhz, stk) else
  if (arg='node.ram')           then Result := pushFloat(MyGPUID.ram, stk) else
  if (arg='node.gflops')        then Result := pushFloat(MyGPUID.gigaflops, stk) else
  if (arg='node.issmp')         then Result := pushBool(MyGPUID.issmp, stk) else
  if (arg='node.isht')          then Result := pushBool(MyGPUID.isht, stk) else
  if (arg='node.is64bit')       then Result := pushBool(MyGPUID.is64bit, stk) else
  if (arg='node.iswine')        then Result := pushBool(MyGPUID.iswineemulator, stk) else
  if (arg='node.isscreensaver') then Result := pushBool(MyGPUID.isrunningasscreensaver, stk) else
  if (arg='node.cpus')          then Result := pushFloat(MyGPUID.nbcpus, stk) else
  if (arg='node.uptime')        then Result := pushFloat(MyGPUID.uptime, stk) else
  if (arg='node.totuptime')     then Result := pushFloat(MyGPUID.totaluptime, stk) else
  if (arg='node.cputype')       then Result := pushStr(MyGPUID.cputype, stk) else
  if (arg='node.localip')       then Result := pushStr(MyGPUID.localip, stk) else
  if (arg='node.longitude')     then Result := pushFloat(MyGPUID.longitude, stk) else
  if (arg='node.latitude')      then Result := pushFloat(MyGPUID.latitude, stk) else
    raise Exception.Create('Node argument '+QUOTE+arg+QUOTE+' not registered in specialcommands.pas');
  Result := true;
end;

function TSpecialCommand.execThreadCommand(arg : String; var stk : TStack) : boolean;
var float : TStkFloat;
begin
  Result := false;
  if (arg='thread.sleep') then
       begin
         Result := popFloat(float, stk);
         if Result then Sleep(Round(float * 1000));
       end
      else
    raise Exception.Create('Thread argument '+QUOTE+arg+QUOTE+' not registered in specialcommands.pas');      
end;

function TSpecialCommand.execCoreCommand(arg : String; var stk : TStack) : boolean;
begin
  Result := false;  
  if (arg='core.threads')        then Result := pushFloat(TMCompStatus.threads, stk) else
  if (arg='core.maxthreads')     then Result := pushFloat(TMCompStatus.maxthreads, stk) else
  if (arg='core.downloads')      then Result := pushFloat(TMDownStatus.threads, stk) else
  if (arg='core.maxdownloads')   then Result := pushFloat(TMDownStatus.maxthreads, stk) else
  if (arg='core.services')       then Result := pushFloat(TMServiceStatus.threads, stk) else
  if (arg='core.maxservices')    then Result := pushFloat(TMServiceStatus.maxthreads, stk) else
  if (arg='core.isidle')         then Result := pushBool(TMCompStatus.isIdle, stk) else
  if (arg='core.hasresources')   then Result := pushBool(TMCompStatus.hasResources, stk) else
  if (arg='core.version')        then Result := pushStr(CORE_VERSION, stk) else
  if (arg='core.stkversion')     then Result := pushStr(STACK_VERSION, stk) else
  if (arg='core.registeredjobs') then Result := frontman_.getStandardQueue().getRegisteredList(stk) else
    raise Exception.Create('Core argument '+QUOTE+arg+QUOTE+' not registered in specialcommands.pas');
  Result := true;   
end;

function TSpecialCommand.execPluginCommand(arg : String; var stk : TStack) : boolean;
var str, field   : TStkString;
    pluginName   : String;
    plugin       : TPlugin;
begin
  Result := false;
  if (arg='plugin.list') then  
          begin
            Result := plugman_.getPluginList(stk);
            Exit;
          end;
  
  // all other commands have a string as argument
  Result := popStr(str, stk);
  if not Result then Exit;
  
  if (arg='plugin.load') then      Result := plugman_.loadOne(str, stk.error) else
  if (arg='plugin.discard') then   Result := plugman_.discardOne(str, stk.error) else
  if (arg='plugin.isloaded') then  Result := pushBool(plugman_.isLoaded(str), stk) else
  // tells which plugin implements a given function
  if (arg='plugin.which') then
        begin  
           Result := plugman_.method_exists(str, pluginName, stk.error);
           pushStr(pluginName, stk);
        end
  else      
  if (arg='plugin.isable') then
        begin  
          Result := pushBool(plugman_.method_exists(str, pluginName, stk.error), stk);
        end
  else
  if (arg='plugin.getfield') then
        begin
          pluginName := str;
          Result := popStr(field, stk);
          if not Result then Exit;
          plugin := plugman_.getPlugin(pluginName);
          if plugin<>nil then
                begin
                  str := plugin.getDescription(field);
                  Result := pushStr(str, stk);
                end
              else
                begin
                  stk.error.errorId  := COULD_NOT_FIND_PLUGIN_ID;
                  stk.error.errorMsg := COULD_NOT_FIND_PLUGIN;
                  stk.error.errorArg := pluginName;
                end;
        end
  else
    raise Exception.Create('Plugin argument '+QUOTE+arg+QUOTE+' not registered in specialcommands.pas');
end;

function TSpecialCommand.execResultCommand(arg : String; var stk : TStack) : boolean;
var coll  : TResultCollection;
    jobId : String;
    i     : Longint;
begin
   Result := false;
   Result := popStr(jobId, stk);
   if not Result then Exit;

   Result := rescollector_.getResultCollection(jobId, coll);
   if (not Result) or (coll.idx = 0) then
        begin
		  stk.error.errorId  := STILL_NO_RESULTS_ID;
		  stk.error.errorMsg := STILL_NO_RESULTS;
		  stk.error.errorArg := '(JobId: '+jobId+')';
		  Exit;
		end;
		
   if (arg='result.last') then
       begin
	      if coll.isFloat[coll.idx] then
		    pushFloat(coll.resFloat[coll.idx], stk)
		  else
                    pushStr(coll.resStr[coll.idx], stk);
       end
   else
   if (arg='result.first') then
       begin
	      if coll.isFloat[1] then
		    Result := pushFloat(coll.resFloat[1], stk)
		  else
                    Result := pushStr(coll.resStr[1], stk);
	   end
   else 
   if (arg='result.history') then
       begin
	     for i:=1 to coll.idx do 
		   Result := pushStr(coll.resStr[i], stk);
	   end
   else	   
   if (arg='result.avg')      then Result := pushFloat(coll.avg, stk) else
   if (arg='result.n')        then Result := pushFloat(coll.N, stk) else
   if (arg='result.nfloat')   then Result := pushFloat(coll.N_float, stk) else
   if (arg='result.sum')      then Result := pushFloat(coll.sum, stk) else
   if (arg='result.min')      then Result := pushFloat(coll.min, stk) else
   if (arg='result.max')      then Result := pushFloat(coll.max, stk) else
   if (arg='result.stddev')   then Result := pushFloat(coll.stddev, stk) else
   if (arg='result.variance') then Result := pushFloat(coll.variance, stk) else
   if (arg='result.overrun')  then Result := pushBool(coll.overrun, stk)
  else
    raise Exception.Create('Result argument '+QUOTE+arg+QUOTE+' not registered in specialcommands.pas');
  
end;

function TSpecialCommand.execFrontendCommand(arg : String; var stk : TStack) : boolean;
var 
  broadcast : TRegisterQueue;
  regInfo   : TRegisterInfo;
  types     : TStkTypes;
  
  jobId, 
  IP, 
  path, filename, 
  executable, form, fullname : TStkString;
  port : TStkFloat;
  
begin
  Result := false;
  broadcast := frontman_.getBroadcastQueue();
  if (arg='frontend.udp.register') then
	     begin
		   types[1]:= STRING_STKTYPE;   types[2]:= STRING_STKTYPE;
		   types[3]:= FLOAT_STKTYPE;    types[4]:= STRING_STKTYPE;
		   types[5]:= STRING_STKTYPE;   types[6]:= STRING_STKTYPE;
		   Result := typeOfParametersCorrect(6, stk,  types);
		   if Result then 
		     begin
			   popStr(fullname   , stk);
			   popStr(form       , stk);
			   popStr(executable , stk);
			   popFloat(port     , stk);
			   popStr(IP         , stk);
			   popStr(jobId      , stk);
                           regInfo := frontman_.prepareRegisterInfo4UdpFrontend(jobId, IP, Round(port), executable, form, fullname);
			   broadcast.registerJob(regInfo);
			 end;  
		 end
    else
	  if (arg='frontend.files.register') then
	     begin
		   types[1]:= STRING_STKTYPE;   types[2]:= STRING_STKTYPE;
		   types[3]:= STRING_STKTYPE;   types[4]:= STRING_STKTYPE;
		   types[5]:= STRING_STKTYPE;   types[6]:= STRING_STKTYPE;
		   Result := typeOfParametersCorrect(6, stk,  types);
		   if Result then 
		     begin
			   popStr(fullname   , stk);
			   popStr(form       , stk);
			   popStr(executable , stk);
			   popStr(path       , stk);
			   popStr(filename   , stk);
			   popStr(jobId      , stk);
		           regInfo := frontman_.prepareRegisterInfo4FileFrontend(jobId, path, filename, executable, form, fullname);
			   broadcast.registerJob(regInfo);
			 end;  
		 end
    else
	if (arg='frontend.unregister') then
	   begin
	     types[1]:= STRING_STKTYPE;   types[2]:= STRING_STKTYPE;
		 Result := typeOfParametersCorrect(2, stk,  types);
		 if Result then 
		      begin
                            popStr(form, stk);
			    popStr(jobId, stk);
			    broadcast.unregisterJob(jobId, form);
                      end;
	   end
 	else
    if (arg='frontend.list') then
       Result := broadcast.getRegisteredList(stk)
    else	   
    raise Exception.Create('Frontend argument '+QUOTE+arg+QUOTE+' not registered in specialcommands.pas');
  Result := true;	
end;

end.
