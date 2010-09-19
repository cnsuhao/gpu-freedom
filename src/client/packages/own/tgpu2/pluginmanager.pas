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

uses SysUtils, stacks, plugins;

const MAX_PLUGINS = 256;  // how many plugins we can load at maximum
      MAX_HASH    = 512;  // how many function calls we hash for faster retrieval

type THashPlugin = record
     method   : String;
     callplug : PPlugin;
end;
      
type
  TPluginManager = class(TObject)
   public  
    constructor Create(Path, Extension : String);
    destructor  Destroy();
    
    procedure loadAll();
    procedure discardAll();
    function  loadOne(pluginName : String)  : Boolean;
    function  discardOne(pluginName : String)  : Boolean;
    
    // calls the method, and passes the stack to the method
    function method_execute(name : String, var Stk : TStack) : Boolean;
 
    
   private
     path_, extension    : String;
     plugs_              : array[1..MAX_PLUGINS] of PPlugins;
     hash_               : array[1..MAX_HASH] of THashPlugin;
     plugsidx_, hashidx_ : Longint;  // indexes for the arrays above
     function  load(pluginName : String)  : Boolean;
    
end;



implementation

constructor TPluginManager.Create(Path, Extension : String);
var i : Longint;
begin
  inherited Create();
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
end;

procedure TPluginManager.loadAll();
var retval  : integer;
    SRec:   TSearchRec;
begin
  if plugidx_<>0 then raise Exception.Create('Please call discardAll() first');

  retval := FindFirst(path_+SEPARATOR+'*.' + extension_, faAnyFile, SRec);
  while retval = 0 do
   begin
     if (SRec.Attr and (faDirectory or faVolumeID)) = 0 then
        load(SRec.Name);
   end;
end;

procedure TPluginManager.discardAll();
var i : Longint;
begin
 for i:=1 to plugidx_ do
  begin
   plugs_[i]^.discard();
   plugs_[i]^.Free;
  end; 
 plugidx_ := 0;
end;

function TPluginManager.method_execute(name : String, var Stk : TStack) : Boolean;
var i : Longint;
begin
 Result := False;
 if Trim(name)='' then Exit;
 // check if we have the method call in the hash table
 for i:=1 to MAX_HASH do
      if (hash_[i].method=name) then
         begin
           if hash_[i].callplug^.isloaded then
              Result := hash_[i].callplug^.method_execute(name, Stk);
           Exit;   
         end;
    
 // go through the list and call the method, register to hash if we found the plugin
 for i:=1 to plugidx_ do
     begin
       if (plugs_[i]^.isloaded() and plugs_[i]^.method_exists(name)) then
          begin
            Result := plugs_[i]^.method_execute(name, Stk);
            
            // remember on the hash where we found the function call
            Inc(hashidx_);
            if (hashidx_>MAX_HASH) then hashidx_ := 1;
            hash_[i].method   := name;
            hash_[i].callplug := plugs_[i];
            Exit;
          end;
     end;
  
  // we did not find the method, we report it as an error
  Stk.error.ErrorID := METHOD_NOT_FOUND_ID;
  Stk.error.ErrorMsg := METHOD_NOT_FOUND;
  Stk.error.ErrorArg := name;  
end;

function  TPluginManager.loadOne(pluginName : String)  : Boolean;
var i : Longint;
    plug : TPlugin;
begin
 // we check first if the plugin is already loaded once
 for i :=1 to plugidx_ do
    if plugs_[i]^.getName() = pluginName then
       begin
         if (not plugs_[i]^.isloaded()) then
            begin           
             Result := plugs_[i].load();
             Exit;
            end; 
       end;
  
  // this plugin is new then
  Result := load(pluginName);
end;

function  TPluginManager.discardOne(pluginName : String);
var i : Longint;
begin
 Result := false;
 // we check first if the plugin is already loaded once
 for i :=1 to plugidx_ do
    if plugs_[i]^.getName() = pluginName then
       begin
         if (plugs_[i]^.isloaded()) then
            Result := plugs_[i]^.discard();
       end;
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