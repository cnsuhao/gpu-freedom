unit managedthreads;

{$mode objfpc}{$H+}

interface

uses
{$ifdef unix}
   cthreads,
 {$endif}
  Classes, SysUtils;

type TManagedThread = class(TThread)
 public
   constructor Create();
   function    isDone()     : Boolean;
   function    isErroneus() : Boolean;

 protected
   done_,
   erroneous_ : Boolean;
end;

implementation

constructor TManagedThread.Create();
begin
 inherited Create(false);
 done_ := false;
 erroneous_ := false;
end;

function  TManagedThread.isDone()     : Boolean;
begin
 Result := done_;
end;


function  TManagedThread.isErroneus() : Boolean;
begin
 Result := erroneous_;
end;


end.

