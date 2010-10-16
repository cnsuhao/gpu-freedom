unit pluginmanagers;
{
 The PluginManager handles a linked list of Plugins, which are DLLs
 containing computational algorithms. DLLs are a collection of
 method calls, each call contains a function with the computational
 algorithm. The function calls all have the same signature defined
 in Definitons class.
 
 PluginManager can load and unload them dinamycally at runtime.
 It can find functions inside the DLL, and execute them.
 It can retrieve a list of loaded plugins and tell if a particular
 plugin is loaded or not. 
 
  (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM
  This unit is released under GNU Public License (GPL)
}

interface

uses SysUtils, SyncObjs,
     stacks, plugins, stkconstants, loggers, utils;

const MAX_PLUGINS = 128;  // how many plugins we can load at maximum
      MAX_HASH    = 64;  // how many function calls we hash for faster retrieval

type THashPlugin = record
     method,
     plugname : String;
     callplug : TPlugin;
end;
      
type
  TPluginManager = class(TObject)
   public  
    constructor Create(Path, Extension : String; logger : TLogger);
    destructor  Destroy();
    
    procedure loadAll(var error : TStkError);
    procedure discardAll();
    function  loadOne(pluginName : String; var error : TStkError)  : Boolean;
    function  discardOne(pluginName : String; var error : TStkError)  : Boolean;
    function  isLoaded(pluginName : String)  : Boolean;
    function  getPluginList(var stk : TStack) : Boolean;
    function  getPlugin(name : String) : TPlugin;

    // calls the method, and passes the stack to the method
    function method_execute(name : String; var stk : TStack) : Boolean;
    // checks if the method exists, returns the plugin name if found
    function method_exists(name : String; var plugName : String; var error : TStkError) : Boolean;
    
   private
     path_, extension_   : String;
     plugs_              : array[1..MAX_PLUGINS] of TPlugin;
     hash_               : array[1..MAX_HASH] of THashPlugin;
     plugidx_, hashidx_  : Longint;  // indexes for the arrays above

     CS_ : TCriticalSection;
     logger_ : TLogger;

     procedure clearHash;
     function  load(pluginName : String;  var error : TStkError)  : Boolean;
     procedure register_hash(funcName : String; var plugin : TPlugin);
     procedure addNotFoundError(name : String; var error : TStkError);
     function  retrievePlugin(funcname : String; var plugname : String; var p : TPlugin; var error : TStkError) : Boolean;
     
end;



implementation

constructor TPluginManager.Create(Path, Extension : String; logger : TLogger);
begin
  inherited Create();
  CS_ := TCriticalSection.Create;
  logger_ := logger;
  path_ := Path;
  extension_ := Extension;
  plugidx_ := 0;
  clearHash();
end;


destructor  TPluginManager.Destroy();
var i : Longint;
begin
  discardAll();
  CS_.free;
end;

procedure TPluginManager.clearHash();
var i : Longint;
begin
  for i :=1 to MAX_PLUGINS do plugs_[i] := nil;
  for i :=1 to MAX_HASH do
     begin
       hash_[i].method := '';
       hash_[i].callplug := nil;
     end;
end;

procedure TPluginManager.loadAll(var error : TStkError);
var retval    : Longint;
    SRec      : TSearchRec;
    filename  : String;
begin
  CS_.Enter;
  if plugidx_<>0 then raise Exception.Create('Please call discardAll() first');

  retval := FindFirst(path_+PathDelim+'*.' + extension_, faAnyFile, SRec);
  while retval = 0 do
   begin
     filename := SRec.Name;
     filename := ExtractParam(filename, '.'); // removing extension
     if (SRec.Attr and (faDirectory or faVolumeID)) = 0 then
        load(filename, error);

     retval := FindNext(SRec);
   end;
  logger_.log('All plugins loaded.');
  CS_.Leave;  
end;

procedure TPluginManager.discardAll();
var i : Longint;
begin
 CS_.Enter;
 for i:=1 to plugidx_ do
  begin
   plugs_[i].discard();
   plugs_[i].Free;
   plugs_[i]:= nil;
  end; 
 plugidx_ := 0;
 clearHash();
 logger_.log('All plugins discarded.');
 CS_.Leave;
end;

function  TPluginManager.isLoaded(pluginName : String)  : Boolean;
var i : Longint;
begin
 CS_.Enter;
 Result := false;
 for i:=1 to plugidx_ do
    begin
     if plugs_[i].isloaded() and
        (plugs_[i].getName()=pluginName) then
	    Result := true;
    end;
 CS_.Leave;
end;

function TPluginManager.getPluginList(var stk : TStack) : Boolean;
var i : Longint;
begin
 CS_.Enter;
 Result := false;
 for i:=1 to plugidx_ do
     if TPlugin(plugs_[i]).isloaded() then
       begin
         if (not pushStr(plugs_[i].getName(), stk)) then
                begin
                  CS_.Leave;
                  Exit;
                end;
              
	   end;
 Result := true;
 CS_.Leave;
end;

function  TPluginManager.getPlugin(name : String) : TPlugin;
var i : Longint;
begin
 CS_.Enter;
 Result := nil;
 for i:=1 to plugidx_ do
     if TPlugin(plugs_[i]).isloaded() then
       begin
         if plugs_[i].getName()=name then
                begin
                  Result := plugs_[i];
                  CS_.Leave;
                  Exit;
                end;

	   end;
 CS_.Leave;
end;

procedure TPluginManager.register_hash(funcName : String; var plugin : TPlugin);
begin
 CS_.Enter;
 // remember on the hash where we found the function call
 Inc(hashidx_);
 if (hashidx_>MAX_HASH) then hashidx_ := 1;
 hash_[hashidx_].method   := funcName;
 hash_[hashidx_].callplug := plugin;
 hash_[hashidx_].plugname := plugin.getName();
 CS_.Leave;
end;


procedure TPluginManager.addNotFoundError(name : String; var error : TStkError);
begin
  // we did not find the method, we report it as an error
  error.ErrorID := METHOD_NOT_FOUND_ID;
  error.ErrorMsg := METHOD_NOT_FOUND;
  error.ErrorArg := name;  
end;

function TPluginManager.retrievePlugin(funcname : String; var plugname : String; var p : TPlugin; var error : TStkError) : Boolean;
var i : Longint;
begin
 Result := false;
 plugName := '';
 if Trim(funcName)='' then Exit;
 
 CS_.Enter;
 // check if we have the method call in the hash table
 for i:=1 to MAX_HASH do
      if (hash_[i].method=funcname) then
         begin
           if hash_[i].callplug.isloaded then
             begin 
			   p := hash_[i].callplug;
			   plugname := hash_[i].plugname;
			   Result := true;
			   CS_.Leave;
                           Exit;
	     end;
         end;
    
 // go through the list and call the method, register to hash if we found the plugin
 for i:=1 to plugidx_ do
     begin
       if (plugs_[i].isloaded() and plugs_[i].method_exists(funcName)) then
          begin
                        register_hash(funcName, plugs_[i]);
			p := plugs_[i];
			plugName :=  plugs_[i].getName();
			Result := true;
                        CS_.Leave;
			Exit;
          end;
     end;
  
 addNotFoundError(funcName, error);
 CS_.Leave;
end;

function TPluginManager.method_execute(name : String; var Stk : TStack) : Boolean;
var plugname : String;
    p        : TPlugin;
begin
 Result := retrievePlugin(name, plugname, p, stk.error);
 if Result then
      Result := p.method_execute(name, stk);
end;

function TPluginManager.method_exists(name : String; var plugName : String; var error : TStkError) : Boolean;
var
    p        : TPlugin;
begin
 p := nil; // not used
 Result := retrievePlugin(name, plugname, p, error);
end; 

function  TPluginManager.loadOne(pluginName : String; var error : TStkError)  : Boolean;
var i : Longint;
    plug : TPlugin;
begin
 CS_.Enter;
 // we check first if the plugin is already loaded once
 for i :=1 to plugidx_ do
    if plugs_[i].getName() = pluginName then
       begin
         if (not plugs_[i].isloaded()) then
            begin           
             Result := plugs_[i].load();
             CS_.Leave;
	     Exit;
            end; 
       end;
  
  // this plugin is new then
  Result := load(pluginName, error);
  if not Result then
       begin
        error.errorID  := COULD_NOT_LOAD_PLUGIN_ID;
        error.errorMsg := COULD_NOT_LOAD_PLUGIN;
        error.errorArg := '('+pluginName+'.'+extension_+')';
       end;
  CS_.Leave;
end;

function TPluginManager.discardOne(pluginName : String; var error : TStkError): Boolean;
var i : Longint;
begin
 CS_.Enter;
 Result := false;
 // we check first if the plugin is already loaded once
 for i :=1 to plugidx_ do
    if (plugs_[i].getName() = pluginName) and
         plugs_[i].isloaded() then
           begin 
              Result := plugs_[i].discard();
              if not Result then
                  begin
                   error.errorID  := COULD_NOT_DISCARD_PLUGIN_ID;
                   error.errorMsg := COULD_NOT_DISCARD_PLUGIN;
                   error.errorArg := '('+pluginName+'.'+extension_+')';
  		   CS_.Leave;
		   Exit;
                  end;
               logger_.log('Plugin '+plugs_[i].getName()+' discarded');
           end;

 Result := true;
 CS_.Leave;	   
end; 

function  TPluginManager.load(pluginName : String; var error : TStkError) : Boolean;
var plug : TPlugin;
    plugVersion : String;
begin
 Result := False;
 plug := TPlugin.Create(path_, pluginName, extension_);
 plug.load();
 if plug.isloaded() then
    begin
     Inc(plugidx_);
     if (plugidx_>MAX_PLUGINS) then 
       begin
        Dec(plugidx_);
        plug.discard();
        error.errorId  := MAX_NUMBER_OF_PLUGINS_REACHED_ID;
        error.errorMsg := MAX_NUMBER_OF_PLUGINS_REACHED;
        error.errorArg := '('+IntToStr(MAX_PLUGINS)+')';
        logger_.log(LVL_SEVERE, 'Maximum number of plugins reached '+error.errorArg);
        Exit;
       end;
     plugVersion := plug.getDescription('stkversion');
     if STACK_VERSION<>plugVersion then
       begin
        error.errorId  := STACK_VERSION_MISMATCH_ID;
        error.errorMsg := STACK_VERSION_MISMATCH;
        error.errorArg := 'v'+Plug.getName()+'='+plugVersion+' vCore='+STACK_VERSION;
        logger_.log(LVL_SEVERE, 'Stack version mismatch in plugin: '+error.errorArg);
        Dec(plugidx_);
        plug.discard();
        Exit;
       end;

     plugs_[plugidx_] := plug;
     logger_.log('Plugin '+plugs_[plugidx_].getName()+' loaded at slot '+IntToStr(plugidx_)+'.');
    end;
  Result := plug.isLoaded();   
end;

end.
