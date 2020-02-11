// Copyright 2016-2020 NVIDIA
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//        http://www.apache.org/licenses/LICENSE-2.0
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#pragma once

template <class T>
class SinglyLinkedList 
{
public:
    struct Node 
    {
        T data;
        Node * next;
    };
    
    Node * head;
    
public:
    SinglyLinkedList();

    void insert(Node * previousNode, Node * newNode);
    void remove(Node * previousNode, Node * deleteNode);
};

template <class T>
SinglyLinkedList<T>::SinglyLinkedList()
{
    
}

template <class T>
void SinglyLinkedList<T>::insert(Node* previousNode, Node* newNode)
{
    if (previousNode == nullptr) 
    {
        // Is the first node
        if (head != nullptr) 
        {
            // The list has more elements
            newNode->next = head;           
        }
        else 
        {
            newNode->next = nullptr;
        }
        head = newNode;
    } 
    else 
    {
        if (previousNode->next == nullptr)
        {
            // Is the last node
            previousNode->next = newNode;
            newNode->next = nullptr;
        }
        else 
        {
            // Is a middle node
            newNode->next = previousNode->next;
            previousNode->next = newNode;
        }
    }
}

template <class T>
void SinglyLinkedList<T>::remove(Node* previousNode, Node* deleteNode)
{
    if (previousNode == nullptr)
    {
        // Is the first node
        if (deleteNode->next == nullptr)
        {
            // List only has one element
            head = nullptr;            
        }
        else 
        {
            // List has more elements
            head = deleteNode->next;
        }
    }
    else 
    {
        previousNode->next = deleteNode->next;
    }
}

