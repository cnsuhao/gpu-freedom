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
    resexpr : Boolean;
begin
  Result := False;
  argRetriever := TArgRetriever.Create(job);
   
  while argRetriever_.hasArguments() do
     begin
       arg := getArgument(error);
       case arg.argType of
            GPU_ERROR :
                   begin
                     argRetriever.Free;
                     Exit; // the error structure contains the error
                   end;  
            GPU_FLOAT : // a float was detected
                   begin
                     Inc(stk.Idx);
                     if maxStackReached(stk, error) then 
                           begin
                             argRetriever.Free;                           
                             Exit;
                           end; 
                     Stk.Stack[stk.Idx] := arg.argvalue;
                     Stk.StrStack[stk.Idx] := '';                     
                   end;
            GPU_STRING : // a string was detected
                   begin
                     Inc(stk.Idx);
                     if maxStackReached(stk, error) then 
                         begin
                           argRetriever.Free;
                           Exit;
                         end;  
                     Stk.Stack[stk.Idx] := INF;
                     Stk.StrStack[stk.Idx] := arg.argstring;                     
                   end;
             GPU_EXPRESSION :
                   begin
                     // we found an expression, we need to recursively call this method
                     resexpr := parse(arg.argstring, stk, error);
                     if (not resexpr) then
                           begin
                             argRetriever.Free;
                             Exit;
                           end;
                   end;                   
              
       
       end; // case
     
     end; // while
   
   argRetriever.Free;
   Result := True;
end;

end.