{$DEFINE MSWINDOWS}
{$IFDEF VER170}
{$DEFINE D7}
{$ENDIF}
{$IFDEF VER150}
{$DEFINE D7}
{$ENDIF}
unit definitions;

// this is a new stack proposed by nanobit
// it is currently not used in GPU


interface

{$IFDEF MSWINDOWS}
uses Windows;

{$ENDIF}

const
  MAXSTACK = 128; {Maximum size of Stack in Virtual Machine}
  MAX_COLLECTING_IDS = 128; {Maximum number of Jobs we keep also average track}
  WRONG_PACKET_ID  = 0.0 / 0.0; {NAN}
  INF      = 1.0 / 0.0;    {infinite to distinguish PChars from floats}
  APOSTROPHE = Chr(39);    {apostrophe is character to separate strings}
  QUOTE    = Chr(39);      {alias for apostrophe, '}

procedure GpuGetMem(var P: Pointer; Size: integer);
procedure GpuReallocMem(var P: Pointer; Size: integer);
procedure GpuFreeMem(var P: Pointer; Size: integer = 0);


type
  PStack = ^TStack;
  TSendCallBack = procedure(stk: PStack); stdcall;
  TSendStack = procedure; stdcall;



  TGpuGetMem = procedure(var P: Pointer; Size: integer); stdcall;
  TGpuReallocMem = procedure(var P: Pointer; Size: integer); stdcall;
  TGpuFreeMem = procedure(var P: Pointer); stdcall;



//prototype freemem
  TDllFreeMem = procedure (P: Pointer); stdcall;

  //prototypes popstack, pushstack, stackavailable

  TPopStackDouble = function (var Stk: TStack): Double; stdcall;
  TPopStackChar = function (var Stk: TStack): PChar; stdcall;

  TStackHasValue = function (var Stk: TStack): LongBool; stdcall;
  TStackHasDouble = function (var Stk: TStack): LongBool; stdcall;
  TStackHasChar = function (var Stk: TStack): LongBool; stdcall;

  TPushStackDouble = function (var Stk: TStack; Value: Double): longbool; stdcall;
  TPushStackChar = function (var Stk: TStack; Value: PChar): longbool; stdcall;

  TStack_proposed = packed record
    //callbacks at beginning
    //so that we can freely alter TStack without
    // being incompatible with plugins that use only these functions:
    PopStackDouble: TPopStackDouble;
    PopStackChar: TPopStackChar;
    StackHasValue: TStackHasValue;
    StackHasDouble : TStackHasDouble;
    StackHasChar: TStackHasChar;
    PushStackDouble: TPushStackDouble;
    PushStackChar: TPushStackChar;

    Stack: array [0..MAXSTACK-1] of double;
    PCharStack: array[0..MAXSTACK-1] of PChar;
    QCharStack: array[0..MAXSTACK-1] of PChar;
    StIdx: longint; {Index on Stack where Operations take place}
    Progress: double;    {indicates plugin progress from 0 to 100}
    My: Pointer;
    hw:     HWND;
    Update: longbool;
    IsGlobal: longbool;
    MultipleResults: longbool;
    SendCallback: TSendCallBack;
    SendStack:    TSendStack;
    Thread: TObject;
    FreeQMem: TDllFreeMem;
//    GpuFreeMem:    TGpuFreeMem;
    Identity: TGPUIdentity;
  end;

type
  TDllFunction = function(var stk: TStack): boolean; stdcall;


const
  WRONG_GPU_COMMAND = 'UNKNOWN_COMMAND';

var
  MyGPUID: TGPUIdentity;


implementation

//maybe move to gpu_utils?
procedure GpuGetMem(var P: Pointer; Size: integer);
begin
  GetMem(P, Size);
end;

procedure GpuReallocMem(var P: Pointer; Size: integer);
begin
  ReallocMem(P, Size);
end;

procedure GpuFreeMem(var P: Pointer; Size: integer = 0);
begin
  if Size = 0 then
    FreeMem(P)
  else
    FreeMem(P, Size);
end;




end.
