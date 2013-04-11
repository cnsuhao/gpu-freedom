unit jobmanagementforms;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  ExtCtrls, ComCtrls, StdCtrls, Spin, jobapis;

type

  { TJobManagementForm }

  TJobManagementForm = class(TForm)
    btnSelectInputWorkunit: TButton;
    btnSubmitJob: TButton;
    cbJobBox: TComboBox;
    cbJobType: TComboBox;
    cbRequireAck: TCheckBox;
    cbTagOutputWorkunit: TCheckBox;
    cgWorkflow: TCheckGroup;
    cbTagInputWorkunit: TCheckBox;
    edtWorkunitInput: TEdit;
    edtWorkunitOutput: TEdit;
    gbWorkunits: TGroupBox;
    lblNumberOfRequests: TLabel;
    lblOutputWorkunitName: TLabel;
    lblWorkunitInput: TLabel;
    lblJobType: TLabel;
    lblJob: TLabel;
    lblJobDefinitionIdDesc: TLabel;
    lblJobDefinitionId: TLabel;
    OpenDialog: TOpenDialog;
    pnCreateJob: TPanel;
    rbGlobal: TRadioButton;
    rbLocal: TRadioButton;
    seNbRequests: TSpinEdit;
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

