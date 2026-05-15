/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_UI_EVENTS_H
#define METAVISION_SDK_UI_EVENTS_H

namespace Metavision {

class InputEvent {
public:
    enum class KeyboardModifier {
        NONE  = 0,
        SHIFT = 1 << 0,
        ALT   = 1 << 1,
        CTRL  = 1 << 2,
    };
    class KeyboardModifiers {
    public:
        KeyboardModifiers(const KeyboardModifier &mod) : mods_(static_cast<int>(mod)) {}
        bool hasModifier(const KeyboardModifier &mod) const {
            return (static_cast<int>(mod) & mods_) || (static_cast<int>(mod) == 0 && mods_ == 0);
        }
        KeyboardModifiers &operator|=(const KeyboardModifiers &mods) {
            mods_ |= mods.mods_;
            return *this;
        }

    private:
        int mods_;
        KeyboardModifiers() {}
        friend class InputEvent;
        friend bool operator<(const InputEvent::KeyboardModifiers &mods1, const InputEvent::KeyboardModifiers &mods2);
        friend bool operator>(const InputEvent::KeyboardModifiers &mods1, const InputEvent::KeyboardModifiers &mods2);
        friend bool operator==(const InputEvent::KeyboardModifiers &mods1, const InputEvent::KeyboardModifiers &mods2);
    };

    InputEvent() : valid_(false) {}
    InputEvent(const KeyboardModifiers &mods) : modifiers_(mods), valid_(true) {}

    KeyboardModifiers modifiers() const {
        return modifiers_;
    }
    bool valid() const {
        return valid_;
    }

private:
    KeyboardModifiers modifiers_;
    bool valid_;
};

inline bool operator>(const InputEvent::KeyboardModifiers &mods1, const InputEvent::KeyboardModifiers &mods2) {
    return mods1.mods_ > mods2.mods_;
}

inline bool operator<=(const InputEvent::KeyboardModifiers &mods1, const InputEvent::KeyboardModifiers &mods2) {
    return !(mods1 > mods2);
}

inline bool operator<(const InputEvent::KeyboardModifiers &mods1, const InputEvent::KeyboardModifiers &mods2) {
    return mods1.mods_ < mods2.mods_;
}

inline bool operator>=(const InputEvent::KeyboardModifiers &mods1, const InputEvent::KeyboardModifiers &mods2) {
    return !(mods1 < mods2);
}

inline bool operator==(const InputEvent::KeyboardModifiers &mods1, const InputEvent::KeyboardModifiers &mods2) {
    return mods1.mods_ == mods2.mods_;
}

inline bool operator!=(const InputEvent::KeyboardModifiers &mods1, const InputEvent::KeyboardModifiers &mods2) {
    return !(mods1 == mods2);
}

inline InputEvent::KeyboardModifiers operator|(const InputEvent::KeyboardModifiers &mods1,
                                               const InputEvent::KeyboardModifiers &mods2) {
    InputEvent::KeyboardModifiers mods(mods1);
    mods |= mods2;
    return mods;
}

class MouseEvent : public InputEvent {
public:
    enum class Type { NONE = 0, MOVE = 1 << 0, PRESS = 1 << 1, RELEASE = 1 << 2, WHEEL = 1 << 3 };

    enum class Button { NONE = 0, LEFT = 1 << 0, RIGHT = 1 << 1, MIDDLE = 1 << 2 };
    class Buttons {
    public:
        Buttons(const Button &but) : buts_(static_cast<int>(but)) {}
        bool hasButton(const Button &but) const {
            return (static_cast<int>(but) & buts_) || (static_cast<int>(but) == 0 && buts_ == 0);
        }
        Buttons &operator|=(const Buttons &buts) {
            buts_ |= buts.buts_;
            return *this;
        }

    private:
        int buts_;
        Buttons() {}
        friend class MouseEvent;
        friend bool operator<(const MouseEvent::Buttons &buts1, const MouseEvent::Buttons &buts2);
        friend bool operator>(const MouseEvent::Buttons &buts1, const MouseEvent::Buttons &buts2);
        friend bool operator==(const MouseEvent::Buttons &buts1, const MouseEvent::Buttons &buts2);
    };

    MouseEvent() : InputEvent() {}
    MouseEvent(const Type &type, float x, float y, const Buttons &buts, const KeyboardModifiers &mods) :
        InputEvent(mods), type_(type), x_(x), y_(y), buttons_(buts) {}

    Type type() const {
        return type_;
    }
    float x() const {
        return x_;
    }
    float y() const {
        return y_;
    }
    Buttons buttons() const {
        return buttons_;
    }

private:
    Type type_;
    float x_;
    float y_;
    Buttons buttons_;
};

inline bool operator>(const MouseEvent::Buttons &buts1, const MouseEvent::Buttons &buts2) {
    return buts1.buts_ > buts2.buts_;
}

inline bool operator<=(const MouseEvent::Buttons &buts1, const MouseEvent::Buttons &buts2) {
    return !(buts1 > buts2);
}

inline bool operator<(const MouseEvent::Buttons &buts1, const MouseEvent::Buttons &buts2) {
    return buts1.buts_ < buts2.buts_;
}

inline bool operator>=(const MouseEvent::Buttons &buts1, const MouseEvent::Buttons &buts2) {
    return !(buts1 < buts2);
}

inline bool operator==(const MouseEvent::Buttons &buts1, const MouseEvent::Buttons &buts2) {
    return buts1.buts_ == buts2.buts_;
}

inline bool operator!=(const MouseEvent::Buttons &buts1, const MouseEvent::Buttons &buts2) {
    return !(buts1 == buts2);
}

inline MouseEvent::Buttons operator|(MouseEvent::Buttons buts1, MouseEvent::Buttons buts2) {
    MouseEvent::Buttons buts(buts1);
    buts |= buts2;
    return buts;
}

class KeyboardEvent : public InputEvent {
public:
    enum class Type {
        PRESS          = 1 << 0,
        RELEASE        = 1 << 1,
        REPEAT         = 1 << 10,
        PRESS_REPEAT   = Type::REPEAT | Type::PRESS,
        RELEASE_REPEAT = Type::REPEAT | Type::RELEASE
    };

    enum class Symbol {
        ESCAPE = 1 << 10,
        TAB,
        BACKTAB,
        BACKSPACE,
        RETURN,
        ENTER,
        INSERT,
        SUPPR,
        PAUSE,
        PRINT,
        HOME,
        END,
        LEFT,
        UP,
        RIGHT,
        DOWN,
        PAGE_UP,
        PAGE_DOWN,
        F1,
        F2,
        F3,
        F4,
        F5,
        F6,
        F7,
        F8,
        F9,
        F10,
        F11,
        F12,
        CTRL,
        ALT,
        SHIFT,
        SPACE,
    };

    KeyboardEvent() : InputEvent() {}
    KeyboardEvent(Type type, int key, const KeyboardModifiers &mods) : InputEvent(mods), type_(type), key_(key) {}

    Type type() const {
        return type_;
    }
    Symbol symbol() const {
        return static_cast<Symbol>(key_);
    }
    int key() const {
        return key_;
    }

private:
    Type type_;
    int key_ = -1;
};

} // namespace Metavision

#endif // METAVISION_SDK_UI_EVENTS_H
