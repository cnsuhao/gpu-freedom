unit jobmanagementforms;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  ExtCtrls, ComCtrls, StdCtrls, jobapis;

type

  { TJobManagementForm }

  TJobManagementForm = class(TForm)
    cbJobBox: TComboBox;
    lblJobType: TLabel;
    lblJob: TLabel;
    lblJobDefinitionIdDesc: TLabel;
    lblJobDefinitionId: TLabel;
    pnCreateJob: TPanel;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure lblJobDefinitionIdDescClick(Sender: TObject);
  private

  public

  end;

var
  JobManagementForm: TJobManagementForm;

implementation

{ TJobManagementForm }

procedure TJobManagementForm.FormCreate(Sender: TObject);
begin
  Visible := True;
end;

procedure TJobManagementForm.FormDestroy(Sender: TObject);
begin
  //
end;

procedure TJobManagementForm.lblJobDefinitionIdDescClick(Sender: TObject);
begin

end;

initialization
  {$I jobmanagementforms.lrs}

end.

