unit pluginmanager
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
}

interface

uses SysUtils, stacks, plugins, SyncObjs;

const MAX_PLUGINS = 128;  // how many plugins we can load at maximum
      MAX_HASH    = 64;  // how many function calls we hash for faster retrieval

type THashPlugin = record
     method,
     plugname : String;
     callplug : PPlugin;
end;
      
type
  TPluginManager = class(TObject)
   public  
    constructor Create(Path, Extension : String);
    destructor  Destroy();
    
    procedure loadAll();
    procedure discardAll();
    function  loadOne(pluginName : String; var error : TGPUError)  : Boolean;
    function  discardOne(pluginName : String; var error : TGPUError)  : Boolean;
	function  isAlreadyLoaded(pluginName : String)  : Boolean;
    
    // calls the method, and passes the stack to the method
    function method_execute(name : String, var Stk : TStack; var error : TGPUError) : Boolean;
    // checks if the method exists, returns the plugin name if found
    function method_exists(name : String; var plugName : String; var error : TGPUError) : Boolean;
    
   private
     path_, extension    : String;
     plugs_              : array[1..MAX_PLUGINS] of PPlugins;
     hash_               : array[1..MAX_HASH] of THashPlugin;
     plugsidx_, hashidx_ : Longint;  // indexes for the arrays above
     function  load(pluginName : String)  : Boolean;
	 procedure register_hash(funcName : String; plugin : PPlugin);
	 procedure addNotFoundError(var error : TGPUError);
	 function  retrievePlugin(funcname : String; var plugname : String; var p : PPlugin; var error : TGPUError) : Boolean;
     
	 CS_ : TCriticalSection;
end;



implementation

constructor TPluginManager.Create(Path, Extension : String);
var i : Longint;
begin
  inherited Create();
  CS_ := TCriticalSection.Create;
  path_ := Path;
  extension_ := Extension;
  plugidx_ := 0;
  for i :=1 to MAX_PLUGINS do plugs_[i] := nil;
  for i :=1 to MAX_HASH do 
     begin
       hash_[i].method := '';
       hash_[i].callplug := nil;
     end;
  
end;


destructor  TPluginManager.Destroy();
var i : Longint;
begin
  discardAll();
  CS_.free;
end;

procedure TPluginManager.loadAll();
var retval  : integer;
    SRec:   TSearchRec;
begin
  CS_.Enter;
  if plugidx_<>0 then raise Exception.Create('Please call discardAll() first');

  retval := FindFirst(path_+SEPARATOR+'*.' + extension_, faAnyFile, SRec);
  while retval = 0 do
   begin
     if (SRec.Attr and (faDirectory or faVolumeID)) = 0 then
        load(SRec.Name);
   end;
  CS_.Leave;  
end;

procedure TPluginManager.discardAll();
var i : Longint;
begin
 CS_.Enter;
 for i:=1 to plugidx_ do
  begin
   plugs_[i]^.discard();
   plugs_[i]^.Free;
  end; 
 plugidx_ := 0;
 CS_.Leave;
end;

function  TPluginManager.isAlreadyLoaded(pluginName : String)  : Boolean;
begin
 Result := false;
 CS_.Enter;
 for i:=1 to plugidx_ do
     if plugs_i[i].plugname=pluginName then
	    Result := true;
 CS_.Leave;
end;

procedure TPluginManager.register_hash(funcName : String; plugin : PPlugin);
begin
 CS_.Enter;
 // remember on the hash where we found the function call
 Inc(hashidx_);
 if (hashidx_>MAX_HASH) then hashidx_ := 1;
 hash_[i].method   := name;
 hash_[i].callplug := plugin;
 hash_[i].plugname := plugin^.getName();
 CS_.Leave;
end;


procedure TPluginManager.addNotFoundError(var error : TGPUError);
begin
  // we did not find the method, we report it as an error
  error.ErrorID := METHOD_NOT_FOUND_ID;
  error.ErrorMsg := METHOD_NOT_FOUND;
  error.ErrorArg := name;  
end;

function TPluginManager.retrievePlugin(funcname : String; var plugname : String; var p : PPlugin; var error : TGPUError) : Boolean;
var i : Longint;
begin
 Result := false;
 if Trim(name)='' then Exit;
 
 CS_.Enter;
 // check if we have the method call in the hash table
 for i:=1 to MAX_HASH do
      if (hash_[i].method=funcname) then
         begin
           if hash_[i].callplug^.isloaded then
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
       if (plugs_[i]^.isloaded() and plugs_[i]^.method_exists(name)) then
          begin
            register_hash(name, plugs_[i]);
			p := plugs_[i];
			plugName :=  plugs_[i]^.getName();
			Result := true;
            CS_.Leave;
			Exit;
          end;
     end;
  
 addNotFound(error : TError);
 CS_.Leave;
end;

function TPluginManager.method_execute(name : String, var Stk : TStack; var error : TGPUError) : Boolean;
var plugname : String;
    p        : PPlugin;
begin
 Result := retrievePlugin(name, plugname, p, error);
 if Result then
      p^.method_execute(name, stk, error);
end;

function method_exists(name : String; var plugName : String; var error : TGPUError) : Boolean;
var plugname : String;
    p        : PPlugin;
begin
 p:=nil; // not used
 Result := retrievePlugin(name, plugname, p, error);
end; 

function  TPluginManager.loadOne(pluginName : String; var error : TGPUError)  : Boolean;
var i : Longint;
    plug : TPlugin;
begin
 CS_.Enter;
 // we check first if the plugin is already loaded once
 for i :=1 to plugidx_ do
    if plugs_[i]^.getName() = pluginName then
       begin
         if (not plugs_[i]^.isloaded()) then
            begin           
             Result := plugs_[i].load();
             CS_.Leave;
			 Exit;
            end; 
       end;
  
  // this plugin is new then
  Result := load(pluginName);
  if not Result then
       begin
        error.errorID  := COULD_NOT_LOAD_PLUGIN_ID;
        error.errorMsg := COULD_NOT_LOAD_PLUGIN;
        error.errorArg := '('+pluginName+'.'+extension_+')';
       end;
  CS_.Leave;
end;

function  TPluginManager.discardOne(pluginName : String; var error : TGPUError);
var i : Longint;
begin
 CS_.Enter;
 Result := false;
 // we check first if the plugin is already loaded once
 for i :=1 to plugidx_ do
    if plugs_[i]^.getName() = pluginName and
         (plugs_[i]^.isloaded()) then
           begin 
              Result := plugs_[i]^.discard();
              if not Result then
                  begin
                   error.errorID  := COULD_NOT_DISCARD_PLUGIN_ID;
                   error.errorMsg := COULD_NOT_DISCARD_PLUGIN;
                   error.errorArg := '('+pluginName+'.'+extension_+')';
                  end;
			  CS_.Leave;
			  Exit;
           end;
 Result := true;
 CS_.Leave;	   
end; 

function  TPluginManager.load(pluginName : String) : Boolean;
var plug : TPlugin;
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
        raise Exception.Create('Maximum number of plugins reached in pluginmanager.pas!');
       end;
     plugs_[plugidx_] := @plug;  
    end;
  Result := plug.isLoaded();   
end;

end.