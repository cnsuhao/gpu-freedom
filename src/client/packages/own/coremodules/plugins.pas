unit plugins;
{
 This unit manages dynamic links libraries and the object TPlugin in plugin.pas
  which encapsulates one of them.
  
  (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM
  This unit is released under GNU Public License (GPL)
}
interface

uses
  SysUtils, stacks, dynlibs;

type
  PPlugin = ^TPlugin;
  
  TPlugin = class(TObject)
   public 
     constructor Create(Path, Name, Extension : String);
     destructor  Destroy();
     
     function load() : Boolean;
     function discard() : Boolean;
     function isloaded() : Boolean;
     
     function getName() : String;
     
     // check if  a method is present in this dll
     function method_exists(name : String) : Boolean;
     // calls the method, and passes the stack to the method
     function method_execute(name : String; var stk : TStack) : Boolean;
     // helper function to retrieve the pointer to the method call
     function method_pointer(name : String) : PDllFunction;
     
     function getDescription(field : String) : TStkString;
 
 protected
     path_,
     name_,
     extension_ : String;
     
     lib_ :  TLibHandle;
     isloaded_ : Boolean;
end;  

implementation

constructor TPlugin.Create(Path, Name, Extension : String);
begin
  inherited Create(); 
  path_ := Path;
  name_ := Name;
  extension_ := Extension;
end;

destructor TPlugin.Destroy();
begin
  if isloaded_ then discard();
  inherited;
end;

function TPlugin.load() : Boolean;
var
   buf      : array [0..1024] of char;
   filename : String;
begin
  lib_ := 0;
  filename := path_+PathDelim+name_+'.'+extension_;
  if FileExists(filename) then 
     lib_ := LoadLibrary(StrPCopy(buf, filename));
  isloaded_ := (lib_<>0);
  Result := isloaded_;
end;

function TPlugin.discard() : Boolean;
begin
 Result := true;
 try
  FreeLibrary(lib_);
  isloaded_ := false;
 except
   Result := false;
 end; 

end;

function TPlugin.isloaded() : Boolean;
begin
  Result := isloaded_;
end;

function TPlugin.method_pointer(name : String) : PDllFunction; 
var
 buf: array [0..144] of char;
begin
  if (not isloaded_) then Result := nil
  else Result := GetProcedureAddress(lib_, StrPCopy(buf, name));
end;

// check if  a method is present in this dll
function TPlugin.method_exists(name : String) : Boolean;
begin
  Result := Assigned(method_pointer(name));
end;

// calls the method, and passes the stack to the method
function TPlugin.method_execute(name : String; var stk : TStack) : Boolean;
var theFunction : PDllFunction;
begin
    theFunction := method_pointer(name);
    if Assigned(theFunction) then
      Result := TDllFunction(theFunction)(stk);
end;     

function TPlugin.getDescription(field : String) : TStkString;
var theFunction : PDescFunction;
    buf         : array [0..144] of char;

begin
  Result := '';
  theFunction := GetProcedureAddress(lib_, StrPCopy(buf, field));
  // get description
  if Assigned(theFunction) then
        Result := TDescFunction(theFunction)();
end;

function TPlugin.getName() : String;
begin
  Result := name_;
end;     

end.
