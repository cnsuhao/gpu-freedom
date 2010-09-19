unit parser;

interface

uses argretrievers, stacks;

type TGPUParser = class(TObject);
 public 
   constructor Create();
   destructor Destroy();
   
   function parse(job : String; var stk : TStack; var error : TError) : Boolean;
 private
   function maxStackReached(var stk : TStack; var error : TError) : Boolean; 
end;


implementation

constructor TGPUParser.Create();
begin
  inherited Create();
end;

destructor TGPUParser.Destroy();
begin
  inherited;
end;

function maxStackReached(var stk : TStack; var error : TError) : Boolean; 
begin
 Result := false;
 if (stk.Idx>MAX_STACK_PARAM) then
       begin
         Result := true;
         error.errorID := TOO_MANY_ARGUMENTS_ID;
         error.errorMsg := TOO_MANY_ARGUMENTS;
         error.errorArg := '';
         Dec(stk.Idx);
       end;
end;

procedure TGPUParser.parse(job : String; var stk : TStack; var error : TError);
var arg    : TGPUArg;
    argRetriever : TArgRetriever;
    resexpr,
    hasErrors    : Boolean;
begin
  Result := False;
  argRetriever := TArgRetriever.Create(job);
  hasErrors := false;
  
  while (argRetriever_.hasArguments() and (not hasErrors)) do
     begin
       arg := getArgument(error);
       case arg.argType of
            GPU_ERROR :  hasErrors := true; // the error structure contains the error
            GPU_FLOAT : // a float was detected
                   begin
                     Inc(stk.Idx);
                     hasErrors := maxStackReached(stk, error) then 
                     if not hasErrors then
                        begin                     
                         Stk.Stack[stk.Idx] := arg.argvalue;
                         Stk.StrStack[stk.Idx] := '';
                        end; 
                   end;
            GPU_STRING : // a string was detected
                   begin
                     Inc(stk.Idx);
                     hasErrors := maxStackReached(stk, error) then 
                     if not hasErrors then
                           begin 
                             Stk.Stack[stk.Idx] := INF;
                             Stk.StrStack[stk.Idx] := arg.argstring;
                           end;  
                   end;
             GPU_EXPRESSION :
                   begin
                     // we found an expression, we need to recursively call this method
                     resexpr := parse(arg.argstring, stk, error);
                     hasErrors := not resexpr;
                   end;                   
       
       end; // case
     
     end; // while
   
   argRetriever.Free;
   Result := not hasErrors;
end;

end.