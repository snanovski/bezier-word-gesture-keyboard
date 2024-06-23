using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.ConstrainedExecution;
using Unity.XR.OpenVR;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.XR;
using UnityEngine.UI;
using UnityEngine.UIElements;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Inputs;


public class CustomWGInteractionScript : MonoBehaviour
{
    private const string USER_SAMPLE_NAME = "user_22";

    public class PhraseSample
    {
        string algorithm;
        string phrase;
        long start_time;
        bool hasStarted = false;
        double wordsCount = 0;
        long end_time;
        List<String> events;
        public PhraseSample(string algorithm, string phrase)
        {
            this.algorithm = algorithm;
            this.phrase = phrase;
            this.wordsCount = (double)(phrase.Split(" ").Select(x => x.Length).Aggregate((x, y) => x + y)) / 5;
            this.start_time = Time();
            events = new List<string>();
            events.Add($"ALGORITHM:{start_time}:{algorithm}");
            events.Add($"PHRASE:{start_time}:{phrase}");
        }

        public long Time()
        {
            return DateTimeOffset.Now.ToUnixTimeMilliseconds();
        }

        public void Start()
        {
            if (hasStarted) return;
            hasStarted = true;
            events.Add($"START:{Time()}");
        }

        public void AddPoints(List<Vector2> pointsList)
        {
            string points = pointsList.Select(x => $"{x.x.ToString(CultureInfo.InvariantCulture)},{x.y.ToString(CultureInfo.InvariantCulture)}").Aggregate((x, y) => x + "," + y);
            events.Add($"POINTS:{Time()}:{points}");
        }

        public void AddWord(string word)
        {
            events.Add($"WORD:{Time()}:{word}");
            end_time = Time();
        }

        public void DeleteWord()
        {
            events.Add($"DELETED:{Time()}");
        }

        public void Export(string filename)
        {
            double minutes = (double)(end_time - start_time) / 60000;
            events.Add($"WPM:{Time()}:{(wordsCount / minutes).ToString(CultureInfo.InvariantCulture)}");
            using (StreamWriter outputFile = new StreamWriter($"./Assets/Data/User/{filename}.txt", true))
            {
                outputFile.WriteLine("===START===");
                foreach (string line in events) {
                    outputFile.WriteLine(line);
                }
                outputFile.WriteLine("===END===");
            }
        }
    }

    public InputActionProperty testButton;
    public InputActionProperty testButton2;
    public WordGestureKeyboard.WordGestureKeyboard keyboardScript;
    public XRBaseController rightController;

    public BoxCollider keyboardBox;

    public Text outputTextField;
    public Text promptTextField;
    public Text promptNumberTextField;
    public UnityEngine.UI.Button nextPromptButton;

    // file for outputting data
    public TextAsset textDataOutput;

    private PhraseSample currentPhraseSampleData = null;

    private XRRayInteractor interactor;
    private bool _isDrawing = false;
    private List<string> macKenziePhraseSet;
    private int _currentPromptNumber = 0;
    private System.Random _random = new System.Random();

    private const int TOTAL_PROMPTS = 16;
    private string _currentPrompt = "";
    private HashSet<string> seenPrompts = new HashSet<string>();


    private void Awake()
    {
        interactor = rightController.GetComponentsInChildren<XRRayInteractor>()[0];
        //interactor = rightController.GetComponent<XRRayInteractor>();
        testButton2.action.performed += startDrawingAction;
        testButton2.action.canceled += endDrawingAction;

        keyboardScript.result.AddListener(Result);
        keyboardScript.deleteEvent.AddListener(Delete);
        keyboardScript.getPointsEvent.AddListener(GetInputPoints);

        LoadPhraseSet();
        /*using (StreamWriter outputFile = new StreamWriter("./Assets/Data/lol.txt", true))
        {
            float f = 1.4895f;
            outputFile.WriteLine($"{f}");
        }*/

        using (StreamWriter outputFile = new StreamWriter($"./Assets/Data/User/{USER_SAMPLE_NAME}.txt", true))
        {
            outputFile.WriteLine("===EXPERIMENT START===");            
        }
    }

    private void GetInputPoints(List<Vector2> pointsList)
    {
        if (currentPhraseSampleData != null)
        {
            currentPhraseSampleData.AddPoints(pointsList);
        }

        /*const string file_location = "./Assets/Data/output_data.txt";
        string points = pointsList.Select(x => $"{x.x.ToString(CultureInfo.InvariantCulture)},{x.y.ToString(CultureInfo.InvariantCulture)}").Aggregate((x, y) => x + "," + y);
        string expectedWord = GetExpectedWord();
        if (expectedWord.Equals("")) return;
        
        //Debug.Log("Points = " + pointsList.Select(x => $"({x.x.ToString(CultureInfo.InvariantCulture)}, {x.y.ToString(CultureInfo.InvariantCulture)})").Aggregate((x, y) => x + ", " + y));

        using (StreamWriter outputFile = new StreamWriter(file_location, true)) 
        {
            outputFile.WriteLine($"{expectedWord}:{points}");
        }*/
    }   
    
    /*private string GetExpectedWord()
    {
        Debug.Log("outputWords = (" + outputTextField.text + ")");
        Debug.Log("outputWords = (" + outputTextField.text.ToLower().Trim() + ")");
        Debug.Log("promptWords = (" + promptTextField.text + ")");
        Debug.Log("promptWords = (" + promptTextField.text.ToLower().Trim() + ")");
        string[] outputWords = outputTextField.text.ToLower().Trim().Split(" ");
        string[] promptWords = promptTextField.text.ToLower().Trim().Split(" ");
        if (outputTextField.text.Equals("")) outputWords = new string[0];
        if (promptWords.Length <= outputWords.Length) return "";
        else return promptWords[outputWords.Length];
        *//*int i = 0;
        while (i < outputWords.Length && i < promptWords.Length && outputWords[i].Equals(promptWords[i])) i++;
        if (i < promptWords.Length)
            return promptWords[i];
        else
            return "";*//*
    }*/

    private void LoadPhraseSet()
    {
        macKenziePhraseSet = new List<string>();
        foreach (var line in File.ReadLines("Assets/MacKenziePhraseSet.txt"))
        {
            macKenziePhraseSet.Add(line.ToLower());
        }
        Debug.Log("All " + macKenziePhraseSet.Count + " MacKenzie Phrases Loaded!");
    }

    private void startDrawingAction(InputAction.CallbackContext ctx)
    {
        //if (interactor.TryGetCurrent3DRaycastHit(out RaycastHit hit) && hit.rigidbody == keyboardBox)
        //{
            _isDrawing = true;
        //}


    }

    private void endDrawingAction(InputAction.CallbackContext ctx)
    {
        _isDrawing = false;
        keyboardScript.DrawWord(null, false);
    
    }

    [Serializable]
    public class ComplexInteractionEvent : UnityEvent<Transform, bool>
    {
    }

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (_isDrawing)
        {
            if (interactor.TryGetCurrent3DRaycastHit(out RaycastHit hit))
            {
                if (currentPhraseSampleData != null)
                {
                    currentPhraseSampleData.Start();
                }

                Transform hitTransform = new GameObject("HitPoint").transform;
                hitTransform.position = hit.point;
                hitTransform.rotation = Quaternion.identity;

                //keyboardScript.DrawWord(hitTransform, true);
                keyboardScript.DrawWord(rightController.transform, true);
            }
        }
    }

    public void Test(string msg = "frog")
    {   
        Debug.Log(msg);
    }

    void Result(string msg)
    {
        if (msg.Equals(" ")) return;
        // Debug.Log("RESULT ADDED = " + msg);
        if (currentPhraseSampleData != null)
        {
            currentPhraseSampleData.AddWord(msg);
        }

        if (msg.Equals(" "))
        {
            outputTextField.text += " ";
        }
        else
        {
            outputTextField.text += msg + " ";
        }
        //var points = keyboardScript.getUIH().GetTransformedPoints(false);
        //Debug.Log("Points = " + points.Select(x => $"({x.x}, {x.y})").Aggregate((x, y) => x + ", " + y));
        UpdateNextPromptButton();
    }

    void Delete()
    {
        //Debug.Log("DELETE INVOKED");
        if (currentPhraseSampleData != null)
        {
            currentPhraseSampleData.DeleteWord();
        }

        string s = outputTextField.text;
        //s = s.TrimEnd(' ');
        s = s.Substring(0, s.Length - 1);
        int lastSpace = s.LastIndexOf(' ');
        Debug.Log("Last space = " + lastSpace);
        if (lastSpace == -1)
        {
            outputTextField.text = "";
        }
        else
        {
            outputTextField.text = s.Substring(0, lastSpace + 1);
        }
        UpdateNextPromptButton();
    }

    public void SetIsDrawing(bool b)
    {
        _isDrawing = b;
    }

    public void NextPrompt()
    {
        _currentPromptNumber++;
        promptNumberTextField.text = $"{_currentPromptNumber} / {TOTAL_PROMPTS}";
        string newPrompt = macKenziePhraseSet[_random.Next(macKenziePhraseSet.Count)];
        while (seenPrompts.Contains(newPrompt))
        {
            newPrompt = macKenziePhraseSet[_random.Next(macKenziePhraseSet.Count)];
        }
        seenPrompts.Add(newPrompt);
        promptTextField.text = newPrompt;
        outputTextField.text = "";
        int algorithmType =1 + _currentPromptNumber % 2;
        keyboardScript.setCurrentGraphs(algorithmType);
        string algName;
        switch (algorithmType)
        {
            case 2:
                algName = "Cubic Bezier";
                break;
            default:
                algName = "Linear";
                break;
        }
        currentPhraseSampleData = new PhraseSample(algName, newPrompt);
        //keyboardScript.setCurrentGraphs(_currentPromptNumber);
        // do other stuff with timers or whatever is needed
        UpdateNextPromptButton();
    }

    void UpdateNextPromptButton()
    {
        nextPromptButton.interactable =
            _currentPromptNumber == 0 ||
            (outputTextField.text.Trim().Equals(promptTextField.text) && _currentPromptNumber <= TOTAL_PROMPTS);

        if (outputTextField.text.Trim().Equals(promptTextField.text) && currentPhraseSampleData != null)
        {
            currentPhraseSampleData.Export(USER_SAMPLE_NAME);
            currentPhraseSampleData = null;
        }
    }
}
